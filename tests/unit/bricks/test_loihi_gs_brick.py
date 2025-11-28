import networkx as nx
from fugu.bricks.loihi_gs_brick import LoihiGSBrick
from fugu.scaffold import Scaffold


def test_loihi_gs_preprocessing_and_mapping():
    """Test preprocessing and mapping with graph reversal.
    
    Original graph: A -> B (cost 3), A -> C (cost 1), B -> C (cost 2)
    After reversal: B -> A (cost 3), C -> A (cost 1), C -> B (cost 2)
    
    In reversed graph:
    - Node B has fanout 1 (B -> A with cost 3) - NO auxiliary needed (fanout == 1)
    - Node C has fanout 2 (C -> A with cost 1, C -> B with cost 2)
      - C -> A has cost 1 (OK, no aux needed)
      - C -> B has cost 2 (needs aux: C -> aux -> B)
    
    Expected: 1 auxiliary node for C -> B edge
    """
    adj = {
        'A': [('B', 3), ('C', 1)],
        'B': [('C', 2)],
        'C': []
    }

    brick = LoihiGSBrick(adj, source='A', destination='C', name='testGS')
    
    # Build via scaffold
    scaffold = Scaffold()
    scaffold.add_brick(brick)
    scaffold.lay_bricks()
    G = scaffold.graph

    # After reversal and preprocessing, check neurons exist
    assert 'A' in brick.node_to_neuron
    assert 'B' in brick.node_to_neuron
    assert 'C' in brick.node_to_neuron
    
    neuron_A = brick.node_to_neuron['A']
    neuron_B = brick.node_to_neuron['B']
    neuron_C = brick.node_to_neuron['C']

    # Check auxiliary nodes: only 1 for C -> B
    aux_nodes = [n for n in brick.node_to_neuron.keys() 
                 if isinstance(n, str) and '__aux__' in n]
    assert len(aux_nodes) == 1, f"Expected 1 auxiliary node, got {len(aux_nodes)}: {aux_nodes}"
    assert 'C__aux__B' in aux_nodes[0], f"Expected C__aux__B, got {aux_nodes[0]}"

    # Verify neurons exist in graph
    assert neuron_A in G.nodes
    assert neuron_B in G.nodes
    assert neuron_C in G.nodes
    
    # Forward and backward edges should exist for all edges
    # Each edge (u, v, cost) creates:
    # - Forward edge: u -> v with delay 1
    # - Backward edge: v -> u with delay = cost
    
    forward_edges = [(u, v, d['delay']) for u, v, d in G.edges(data=True) 
                     if d.get('direction') == 'forward']
    backward_edges = [(u, v, d['delay']) for u, v, d in G.edges(data=True) 
                      if d.get('direction') == 'backward']
    
    # All forward edges should have delay 1
    for u, v, delay in forward_edges:
        assert delay == 1, f"Forward edge {u}->{v} should have delay 1, got {delay}"
    
    # Should have both forward and backward edges
    assert len(forward_edges) > 0, "Should have forward edges"
    assert len(backward_edges) > 0, "Should have backward edges"
    
    print(f"Test passed: {len(forward_edges)} forward edges, {len(backward_edges)} backward edges, {len(aux_nodes)} auxiliary nodes")
