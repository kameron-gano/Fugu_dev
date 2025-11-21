import networkx as nx
import pytest

from fugu.bricks.loihi_gs_brick import LoihiGSBrick


def test_adjacency_matrix_input():
    # matrix: 0->1 cost 2, 1->2 cost 1
    mat = [
        [0, 2, 0],
        [0, 0, 1],
        [0, 0, 0],
    ]

    brick = LoihiGSBrick(mat, name='matGS')
    G = nx.DiGraph()
    brick.build(G, None, None, None, None)

    n0 = brick.node_to_neuron[0]
    n1 = brick.node_to_neuron[1]
    n2 = brick.node_to_neuron[2]

    # forward edge delays now 1
    assert G.has_edge(n0, n1)
    assert G[n0][n1]['delay'] == 1
    assert G.has_edge(n1, n2)
    assert G[n1][n2]['delay'] == 1

    # backward delays equal to cost
    assert G.has_edge(n1, n0)
    assert G[n1][n0]['delay'] == 2  # cost 2
    assert G.has_edge(n2, n1)
    assert G[n2][n1]['delay'] == 1  # cost 1





def test_float_costs_quantize_when_allowed():
    adj = {
        'A': [('B', 2.7)],
        'B': []
    }
    # allow rounding
    brick = LoihiGSBrick(adj, name='floatGS', require_integer_costs=False)
    G = nx.DiGraph()
    brick.build(G, None, None, None, None)

    nA = brick.node_to_neuron['A']
    nB = brick.node_to_neuron['B']

    assert G.has_edge(nA, nB)
    assert G[nA][nB]['delay'] == 1
    # cost 2.7 -> round to 3 -> backward delay 3
    assert G.has_edge(nB, nA)
    assert G[nB][nA]['delay'] == 3


def test_disconnected_graph_raises_by_default():
    # two components
    adj = {
        'A': [('B', 1)],
        'C': [('D', 1)],
        'B': [],
        'D': []
    }

    brick = LoihiGSBrick(adj, name='disconnGS')
    G = nx.DiGraph()
    with pytest.raises(ValueError):
        brick.build(G, None, None, None, None)


def test_branching_node_creates_auxiliary_nodes_multiple():
    # Node P with three outgoing edges of various costs
    adj = {
        'P': [('A', 1), ('B', 4), ('C', 3)],
        'A': [], 'B': [], 'C': []
    }
    brick = LoihiGSBrick(adj, name='branchGS')
    G = nx.DiGraph()
    brick.build(G, None, None, None, None)

    # Count auxiliary keys in node_to_neuron
    aux_keys = [k for k in brick.node_to_neuron.keys() if isinstance(k, str) and k.startswith('P__aux__')]
    # Expect two auxiliary nodes for edges with cost>1 (B and C)
    assert len(aux_keys) == 2
