import ast
import logging

import networkx as nx
import numpy as np
from numpy import genfromtxt
from scipy.io import loadmat

from .bricks import Brick, input_coding_types
from .input_bricks import Vector_Input
from ..scaffold import Scaffold
from . import spiking_pde as SpikingPDE

rw_logger = logging.getLogger("NRW")

class DensityRW(Brick):
    """Basic Density Random Walk Circuit
    
    Arguments:
        + timesteps - An integer number of timesteps to run the walk
        + transitions - A dictionary of valid transition edges on the graph {source_node:neighbors}, 
                        neighbors should be a list of tuples (destination_node, probability)
        + init_walkers - A dictionary {start_node: number_of_walkers} to initialize the walk
    
    """
    def __init__(self, 
                 timesteps, 
                 transitions, 
                 init_walkers={(10,):30, (13,): 30}, 
                 name=None, 
                 coding='Raster'):
        super(Brick, self).__init__()
        self.is_built = False
        self.metadata = {}
        self.output_coding = coding
        self.supported_codings = input_coding_types
        self.name = name        
        self.brick_tag = 'DensityRW'
        self.net = None
        self.graph = None
        self.injection = None
        self.transitions = transitions
        self.init_walkers = init_walkers
        self.timesteps = timesteps
    
    def build(self,
              graph,
              metadata,
              control_nodes,
              input_lists,
              input_codings):
        """
        Build Density RW brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - dictionary of lists of auxillary networkx nodes.  Expected keys: 'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats.  All codings are allowed.

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + dictionary of control nodes ('complete')
            + list of output (1 output)
            + list of coding formats of output (Coding matches input coding)
        """      
        #Basic Checks
        if len(input_lists) != 1:
            raise ValueError('Incorrect Number of Inputs.')
        for input_coding in input_codings:
            if input_coding not in self.supported_codings:
                raise ValueError("Unsupported Input Coding. Found: {}. Allowed: {}".format(input_coding,
                                                                                           self.supported_codings))       
        
        #Complete Node
        control_node_list = []
        control_node_name = self.name + '_complete'
        graph.add_node(control_node_name,
                       index=(-1,),
                       decay=0.0,
                       potential=0.0,
                       threshold = float(self.timesteps)
                )
        graph.add_edge(control_nodes[0]['complete'], control_node_name, weight=1.0, delay=1)
        timestep_counter_name = self.name + '_timestep_counter'
        graph.add_node(timestep_counter_name,
                       index=(-2,),
                       decay = 0.0,
                       potential = 0.0,
                       threshold = 0.5
                       )
        graph.add_edge(timestep_counter_name, timestep_counter_name, weight=1.0, delay=1)
        graph.add_edge(timestep_counter_name, control_node_name, weight=1.0, delay=1)
        
        
        #Build the walking graph
        self.net = SpikingPDE.MarkovNetwork(initial_walkers=self.init_walkers,
                            transitions=self.transitions,
                            synchronized=True,
                            log_potential=True, log_spikes=True)
        self.graph = self.net.build()
        self.graph = SpikingPDE.to_graph(self.graph, SpikingPDE.neuron_list)
        
        
        #Prep initial position of walkers
        counter_neurons = [node for node in self.graph.nodes if 'counter' in self.graph.nodes[node]['groups']]
        for init_source in self.init_walkers:
            for node in counter_neurons:
                if str(init_source) in self.graph.nodes[node]['groups']:
                    self.graph.nodes[node]['potential']= -self.init_walkers[init_source]
        
        for (neuron, current) in self.net.injection.get(0, []):
            if neuron is self.net.walker_supervisor:
                self.graph.nodes[neuron.name]['potential'] = current
                
        #Rename nodes
        relabel_dictionary = {}
        for node in self.graph.nodes:
            relabel_dictionary[node] = self.name + '_' + str(node) + '_groups_' + str(self.graph.nodes[node]['groups'])
        self.graph = nx.relabel_nodes(self.graph, relabel_dictionary, copy=False)
        
        #Grab references to output nodes
        output_nodes = [node for node in self.graph.nodes if 'readout' in self.graph.nodes[node]['groups']]
        
        #Write indices for outputs (also bad code)
        for node in output_nodes:
            self.graph.nodes[node]['index'] = ast.literal_eval(self.graph.nodes[node]['groups'][1])

        #Add RW nodes to Fugu graph (This is potentially memory intensive)
        graph.update(self.graph)

        self.is_built = True
        return (graph, self.metadata, [{'complete':control_node_name}], [output_nodes], [self.output_coding])

def load_transitions(mat_file,
                remove_sink_connections=True,
                verbose=1):
    SpikingPDE.reset_neuron_list()
    if '.mat' in mat_file:
        prob_mtx = loadmat(mat_file)
        prob_mtx = prob_mtx[mat_file[:-4]]
    elif '.csv' in mat_file:
        prob_mtx = genfromtxt(mat_file, delimiter=',')
    else:
        raise ValueError("Matrix file incompatible")
    N = np.shape(prob_mtx)
    transitions = {}
    for i in range(N[0]):
        neighbors = []
        p = 0
        for j in range(N[1]):
            prob_i_j = prob_mtx[i,j]
            p += prob_i_j
            if prob_i_j > 0:
                neighbors.append(((j,), prob_mtx[i,j]))
        if verbose>0:
            rw_logger.info("node " + str(i) + " has total probability of " + str(p) )
        if remove_sink_connections and len(neighbors)==1 and neighbors[0]==((i,),1.0):
            if verbose>0:
                rw_logger.info("Removing connections from a sink at " + str((i,)))
            neighbors=[]
        transitions[(i,)] = SpikingPDE.Transition(location=(i,), neighbors=neighbors)
    return transitions


def run_miniapp(transitions_file=None,
                neural_timesteps = 100,
                initial_walkers = [],
                sink_connections=False,
                backend='snn'):
  transitions = load_transitions(transitions_file,
                                 remove_sink_connections = not(sink_connections),
                                 verbose=0)
  scaffold = Scaffold()
  if backend == 'snn':
    from ..backends import snn_Backend
    backend_object = snn_Backend()
  elif backend == 'loihi':
    from ..backends import loihi_Backend
    backend_object = loihi_Backend()
  else:
    raise ValueError("Unsupported backend type.")
  initial_walkers = { (t[0],):t[1] for t in initial_walkers}
  scaffold.add_brick(Vector_Input(np.array([1]),coding='Raster',name='Input0'),'input')
  scaffold.add_brick(DensityRW(neural_timesteps, 
                               transitions,
                               init_walkers = initial_walkers,
                               name = 'DensityRW'),
                    [0],
                    output=True)
  scaffold.lay_bricks()
  
  neuron_number_map = dict()
  rw_neurons = [node for node in scaffold.graph.nodes if ('groups' in scaffold.graph.nodes[node])]
  readout_neurons = [node for node in rw_neurons if ('readout' in scaffold.graph.nodes[node]['groups'])]
  for neuron in readout_neurons:
    neuron_number_map[scaffold.graph.nodes[neuron]['neuron_number']] = scaffold.graph.nodes[neuron]['groups'][1]
  backend_object.compile(scaffold)
  result = backend_object.run(neural_timesteps)
  node_col = []
  for neuron in result['neuron_number'].astype(int):
    node_col.append(neuron_number_map[neuron])
  result['node'] = node_col
  return result
