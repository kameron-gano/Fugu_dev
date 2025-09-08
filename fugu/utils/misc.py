from collections import deque
import re


def CalculateSpikeTimes(circuit, main_key='timestep'):
    """
    Function that determines the initial spikes for the input layer of a fugu circuit.

    Arguments:
        + circuit - Fugu circuit
        + main_key - Whether to use timestep or neuron name as the dictionary key

    Returns:
        + Dictionary of when certain neurons spike (keyed by either timestep or neuron name)

    Exceptions:
        + Raises ValueError if main_key is not 'timestep' or 'neuron_name'
    """
    initial_spikes = {}
    input_nodes = [
        node for node in circuit.nodes
        if ('layer' in circuit.nodes[node]) and (
            circuit.nodes[node]['layer'] == 'input')
    ]
    print("Inside CalculateSpikeTimes", [ circuit.nodes[node]['layer'] for node in circuit.nodes])
    max_steps = 0
    for input_node in input_nodes:
        for timestep, spike_list in enumerate(
                circuit.nodes[input_node]['brick']):
            print("Input node:", input_node, "Timestep:", timestep, "Spike list:", spike_list)
            
            if timestep > max_steps:
                max_steps = timestep
    if main_key == 'timestep':
        for i in range(0, max_steps + 1):
            initial_spikes[i] = deque()
        for input_node in input_nodes:
            for timestep, spike_list in enumerate(
                    circuit.nodes[input_node]['brick']):
                if len(spike_list) > 0:
                    spike_list = [clean_neuron_name(str(x)) for x in spike_list]
                    initial_spikes[timestep].extend(spike_list)
                    print("Initial Spikes", initial_spikes)
                    print("Building timestep:", timestep, " with spikes:", spike_list)
    elif main_key == 'neuron_name':
        for input_node in input_nodes:
            for timestep, spike_list in enumerate(
                    circuit.nodes[input_node]['brick']):
                for neuron in spike_list:
                    if neuron not in initial_spikes:
                        initial_spikes[neuron] = []
                    initial_spikes[neuron].append(timestep)
    else:
        raise ValueError("main_key argument must be 'timestep' or 'neuron_name', not {}".format(main_key))

    print("Inside CalculateSpikeTimes initial_spikes:", initial_spikes)
    return initial_spikes

def clean_neuron_name(name):
    """
    Cleans a neuron name string by removing numpy type annotations inside parentheses.
    Example: "Input-0:(np.int64(0),)" -> "Input-0:(0,)"
    """
    # Replace patterns like np.int64(123) with just 123
    return re.sub(r'np\.\w+\((\-?\d+)\)', r'\1', name)
