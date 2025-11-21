import networkx as nx
from fugu.bricks.loihi_gs_brick import LoihiGSBrick


def test_loihi_gs_preprocessing_and_mapping():
    # Small graph: A -> B (cost 3), A -> C (cost 1), B -> C (cost 2)
    adj = {
        'A': [('B', 3), ('C', 1)],
        'B': [('C', 2)],
        'C': []
    }

    brick = LoihiGSBrick(adj, name='testGS')
    G = nx.DiGraph()
    # call build to mutate G
    brick.build(G, None, None, None, None)

    # Check neurons created for A,B,C and the auxiliary node for A->B
    neuron_A = brick.node_to_neuron['A']
    neuron_B = brick.node_to_neuron['B']
    neuron_C = brick.node_to_neuron['C']

    # There must be an aux node corresponding to A->B created
    aux_nodes = [n for n in brick.node_to_neuron.keys() if isinstance(n, str) and n.startswith('A__aux__')]
    assert len(aux_nodes) == 1
    aux = aux_nodes[0]
    neuron_aux = brick.node_to_neuron[aux]

    # Forward synapses should exist with delay 1 (minimum delay)
    assert G.has_edge(neuron_A, neuron_aux)
    assert G[neuron_A][neuron_aux]['delay'] == 1

    assert G.has_edge(neuron_aux, neuron_B)
    assert G[neuron_aux][neuron_B]['delay'] == 1

    # Backward synapses should have delays equal to original cost
    # For A->aux (cost 1) backward delay is 1
    assert G.has_edge(neuron_aux, neuron_A)
    assert G[neuron_aux][neuron_A]['delay'] == 1

    # For aux->B (cost 2) backward delay is 2
    assert G.has_edge(neuron_B, neuron_aux)
    assert G[neuron_B][neuron_aux]['delay'] == 2

    # Edge A->C had cost 1 (forward delay 1, backward delay 1)
    assert G.has_edge(neuron_A, neuron_C)
    assert G[neuron_A][neuron_C]['delay'] == 1
    assert G.has_edge(neuron_C, neuron_A)
    assert G[neuron_C][neuron_A]['delay'] == 1
