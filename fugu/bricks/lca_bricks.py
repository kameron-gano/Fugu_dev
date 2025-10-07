from fugu.bricks import Brick
from fugu.scaffold.port import PortSpec, ChannelSpec
import numpy as np

class LCABrick(Brick):
    def __init__(self, Phi, lam=0.1, dt=1e-3, tau_syn=1.0, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.Phi = Phi
        self.lam = lam  # Sparsity threshold (lambda)
        self.dt = dt  # Time step
        self.tau_syn = tau_syn  # Synaptic time constant
        self.threshold = threshold  # Spike threshold
        self.supported_codings = ['Raster', 'Undefined']

    @classmethod
    def input_ports(cls) -> dict[str, PortSpec]:
        """LCABrick doesn't require inputs - it generates sparse codes."""
        return {}

    @classmethod  
    def output_ports(cls) -> dict[str, PortSpec]:
        """LCABrick outputs sparse activation codes."""
        port = PortSpec(name='output')
        port.channels['data'] = ChannelSpec(name='data', coding=['Raster'])
        port.channels['complete'] = ChannelSpec(name='complete')
        return {port.name: port}

    def normalize_columns(self, A):
        """ Normalize columns of A to unit norm. """
        norms = np.linalg.norm(A, axis=0)
        norms[norms == 0] = 1.0
        return A / norms
                

    def build2(self, graph, inputs: dict = {}):
        from ..scaffold.port import PortData, ChannelSpec, PortSpec
        
        # Normalize dictionary columns
        Phi = self.normalize_columns(self.Phi)
        N = Phi.shape[1]  # Number of dictionary elements
        M = Phi.shape[0]  # Dimensionality of signals

        # Create complete control node
        complete_node_name = self.generate_neuron_name('complete')
        graph.add_node(complete_node_name,
                       index=-1,
                       threshold=0.0,
                       decay=0.0,
                       p=1.0,
                       potential=0.0)

        # Create S-LCA neurons with proper parameters
        neuron_names = []
        for i in range(N):
            neuron_name = self.generate_neuron_name(f"neuron_{i}")
            neuron_names.append(neuron_name)
            graph.add_node(
                neuron_name,
                index=i,
                threshold=self.threshold,
                reset_voltage=0.0,
                leakage_constant=1.0,  # No leak for S-LCA
                potential=0.0,
                bias=0.0,  # Backend will set proper bias values
                p=1.0,  # Deterministic spiking
                dt=self.dt,
                tau_syn=self.tau_syn,
                lam=self.lam,
                neuron_type='CompetitiveNeuron'  # Specify S-LCA neuron type
            )

        # Build lateral inhibitory connections (W = ΦᵀΦ - I)
        W = Phi.T @ Phi
        np.fill_diagonal(W, 0.0)  # Remove self-connections

        # Add lateral inhibition edges with proper weights
        # When neuron i spikes, it sends inhibitory signals to neurons j
        for i in range(N):
            for j in range(N):
                if i != j and W[j, i] > 0:  # Only add significant connections
                    graph.add_edge(neuron_names[i], neuron_names[j], 
                                 weight=W[i, j],  # Inhibitory weight: i inhibits j with strength W[j,i]
                                 delay=1)

        # Create output port data using the proper pattern
        from ..scaffold.port import PortUtil, ChannelData
        result = PortUtil.make_ports_from_specs(LCABrick.output_ports())
        output_port = result['output']
        
        # Set the data channel neurons
        data_channel = output_port.channels['data']
        data_channel.neurons = neuron_names
        
        # Set the complete channel neurons  
        complete_channel = output_port.channels['complete']
        complete_channel.neurons = [complete_node_name]
        
        return result
