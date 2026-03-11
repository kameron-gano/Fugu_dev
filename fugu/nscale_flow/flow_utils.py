import json
import numpy as np
from matplotlib.path import Path

def left_half_circle_marker(radius=1):
    """Create a left half-circle marker without connecting lines."""
    vertices = []
    for angle in np.linspace(np.pi / 2, 3 * np.pi / 2, num=10):  # Points for left half-circle
        vertices.append((radius * np.cos(angle), radius * np.sin(angle)))

    # Define the marker path without closing it
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 1)
    return Path(vertices, codes)

def right_half_circle_marker(radius=1):
    """Create a right half-circle marker without connecting lines."""
    vertices = []
    for angle in np.linspace(-np.pi / 2, np.pi / 2, num=10):  # Points for right half-circle
        vertices.append((radius * np.cos(angle), radius * np.sin(angle)))

    # Define the marker path without closing it
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 1)
    return Path(vertices, codes)

def fugu_spikes_to_json(num_neurons, matrix):
    matrix = result.to_numpy()
    spike_times = {}
    for i in range(num_neurons):
        spike_times[i] = []
    for i in matrix:
        spike_times[i[1]].append(int(i[0]))

def gen_json(graph, timestamps, path):
    # json file
    network_json = {}
    network_json['ticks'] = timestamps
    network_json['neuron_dict'] = {}
    network_json['synapse_dict'] = {}
    
    for i, neuron in enumerate(sorted(graph.nodes)):
        neuron_info = graph.nodes[neuron]
        network_json['neuron_dict'][str(neuron_info['neuron_number'])] = {"fan-in": [],
                                                                     "fan-out": [],
                                                                     "th": neuron_info['threshold'],
                                                                     "leak": 0.0,
                                                                     "bias": 0.0,
                                                                     "refractory period": 0,
                                                                     "prob": neuron_info['p']
                                                                    }
        if 'leakage_constant' in neuron_info:
            network_json['neuron_dict'][str(neuron_info['neuron_number'])]['leak'] = neuron_info['leakage_constant']
        if 'bias' in neuron_info:
            network_json['neuron_dict'][str(neuron_info['neuron_number'])]['bias'] = neuron_info['bias'] 
        if 'refractory_period' in neuron_info:
            network_json['neuron_dict'][str(neuron_info['neuron_number'])]['refractory_period'] = neuron_info['refractory_period']    
    
    for i, synapse in enumerate(graph.edges):
        pre_neuron = graph.nodes[synapse[0]]['neuron_number']
        post_neuron = graph.nodes[synapse[1]]['neuron_number']
        network_json['neuron_dict'][str(post_neuron)]['fan-in'].append(pre_neuron)
        network_json['neuron_dict'][str(pre_neuron)]['fan-out'].append(post_neuron)
        if str(post_neuron) not in network_json['synapse_dict']:
            network_json['synapse_dict'][str(post_neuron)] = {}
        if str(pre_neuron) not in network_json['synapse_dict'][str(post_neuron)]:
            network_json['synapse_dict'][str(post_neuron)][str(pre_neuron)] = {}
        network_json['synapse_dict'][str(post_neuron)][str(pre_neuron)][str(len(network_json['synapse_dict'][str(post_neuron)][str(pre_neuron)]))] = {'fixed-wt': graph.edges[synapse]['weight'],
                                                                                                                                                      'delay': graph.edges[synapse]['delay']}
    with open(path, 'w') as f:
        json.dump(network_json, f) 