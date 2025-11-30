#!/usr/bin/env python3
"""
Comprehensive verification suite for Loihi graph search fanout preprocessing and delay mapping.

This script validates that:
1. Nodes with fanout > 1 have all outgoing edges with cost=1
2. Auxiliary nodes preserve total backward delay correctly
3. Delays are mapped correctly: delay_fugu = cost (compensates for min delay=1)
4. Complex multi-fanout scenarios work correctly
5. Edge cases (fanout=1, cost=1, chains of auxiliary nodes)
"""

from fugu.bricks.loihi_gs_brick import LoihiGSBrick
from fugu.scaffold.scaffold import Scaffold
from collections import defaultdict
import pytest

def test_fanout_preprocessing():
    """Test that fanout constraint is properly enforced."""
    
    # Test case: Node 0 has fanout=2 with costs > 1
    test_adj = {
        0: [(1, 5), (2, 3)],  # fanout=2, both costs > 1 (requires aux nodes)
        1: [(3, 2)],           # fanout=1
        2: [(3, 1)],           # fanout=1, cost=1
        3: []                  # destination
    }
    
    print("="*70)
    print("TEST: Fanout Preprocessing")
    print("="*70)
    print("\nInput adjacency list:")
    for node, neighbors in test_adj.items():
        print(f"  Node {node}: {neighbors}")
    
    # Create brick and check preprocessing
    brick = LoihiGSBrick(test_adj, source=0, destination=3, name='TestBrick')
    nodes, edges = brick._parse_input()
    
    print(f"\nParsed edges:")
    fanout_original = defaultdict(list)
    for u, v, c in edges:
        print(f"  {u} -> {v} (cost={c})")
        fanout_original[u].append((v, c))
    
    print(f"\nOriginal fanout counts:")
    for node in sorted(fanout_original.keys()):
        edges_out = fanout_original[node]
        print(f"  Node {node}: fanout={len(edges_out)}, costs={[c for v,c in edges_out]}")
    
    # Apply preprocessing
    proc_nodes, proc_edges = brick._preprocess_fanout(nodes, edges)
    
    print(f"\n{'─'*70}")
    print("After fanout preprocessing:")
    print(f"{'─'*70}")
    print(f"\nProcessed nodes ({len(proc_nodes)} total):")
    print(f"  Original: {[n for n in proc_nodes if '__aux__' not in str(n)]}")
    print(f"  Auxiliary: {[n for n in proc_nodes if '__aux__' in str(n)]}")
    
    print(f"\nProcessed edges:")
    fanout_processed = defaultdict(list)
    for u, v, c in proc_edges:
        print(f"  {u} -> {v} (cost={c})")
        fanout_processed[u].append((v, c))
    
    # Verify fanout constraint
    print(f"\n{'─'*70}")
    print("VERIFICATION: Fanout Constraint")
    print(f"{'─'*70}")
    violations = []
    for node in proc_nodes:
        edges_out = fanout_processed[node]
        if len(edges_out) > 1:
            costs = [c for v, c in edges_out]
            all_one = all(c == 1 for c in costs)
            status = "✓ PASS" if all_one else "✗ FAIL"
            print(f"  Node {node}: fanout={len(edges_out)}, costs={costs} {status}")
            if not all_one:
                violations.append(node)
    
    assert len(violations) == 0, f"Nodes {violations} violate fanout constraint!"
    print(f"\n✓ PASSED: All nodes with fanout > 1 have cost=1 on all outgoing edges")
    
    # Verify total delay preservation
    print(f"\n{'─'*70}")
    print("VERIFICATION: Total Backward Delay Preservation")
    print(f"{'─'*70}")
    
    # For edge 0->1 (original cost=5):
    # After split: 0->aux (cost=1) + aux->1 (cost=5)
    # Expected backward delays (in Loihi): 0 + 4 = 4 (matches original 5-1=4)
    # In Fugu: 1 + 5 = 6 (both shifted by +1)
    
    print("\nOriginal edge: 0 -> 1 (cost=5)")
    print("  Loihi backward delay: 5 - 1 = 4")
    print("\nAfter preprocessing:")
    aux_0_1 = next((v for v, c in fanout_processed[0] if '__aux__' in str(v) and any(dst == 1 for dst, _ in fanout_processed.get(v, []))), None)
    if aux_0_1:
        cost_0_aux = next(c for v, c in fanout_processed[0] if v == aux_0_1)
        cost_aux_1 = next(c for v, c in fanout_processed[aux_0_1] if v == 1)
        print(f"  Split into: 0 -> {aux_0_1} (cost={cost_0_aux}) + {aux_0_1} -> 1 (cost={cost_aux_1})")
        print(f"  Loihi backward delays: ({cost_0_aux}-1) + ({cost_aux_1}-1) = {cost_0_aux-1} + {cost_aux_1-1} = {cost_0_aux+cost_aux_1-2}")
        print(f"  Fugu delays (cost): {cost_0_aux} + {cost_aux_1} = {cost_0_aux + cost_aux_1}")
        
        # With c-1 formula: 0->aux (cost 1) + aux->1 (cost 4)
        # Total cost: 1 + 4 = 5 (original preserved)
        # Loihi delays: (1-1) + (4-1) = 0 + 3 = 3
        total_loihi_delay = cost_0_aux + cost_aux_1 - 2
        assert total_loihi_delay == 3, f"Expected total Loihi delay = 3, got {total_loihi_delay}"
        assert cost_0_aux + cost_aux_1 == 5, f"Expected total cost = 5, got {cost_0_aux + cost_aux_1}"
        print("  ✓ PASSED: Total cost preserved!")

def test_delay_mapping():
    """Test that delays are correctly mapped to Fugu graph."""
    
    print(f"\n{'='*70}")
    print("TEST: Delay Mapping to Fugu Graph")
    print(f"{'='*70}")
    
    # Simple test case
    test_adj = {
        0: [(1, 3)],  # Single edge with cost=3
        1: []
    }
    
    brick = LoihiGSBrick(test_adj, source=0, destination=1, name='DelayTest')
    scaffold = Scaffold()
    scaffold.add_brick(brick, output=True)
    scaffold.lay_bricks()
    graph = scaffold.graph
    
    print(f"\nInput: 0 -> 1 (cost=3)")
    print(f"Expected:")
    print(f"  Forward: delay=1 (fixed)")
    print(f"  Backward: delay=3 (in Fugu, equivalent to Loihi's 3-1=2, shifted by +1)")
    
    n0 = brick.node_to_neuron[0]
    n1 = brick.node_to_neuron[1]
    
    # Check forward edge
    forward_delay = graph[n0][n1]['delay']
    print(f"\nActual:")
    print(f"  Forward ({n0} -> {n1}): delay={forward_delay}")
    
    # Check backward edge
    backward_delay = graph[n1][n0]['delay']
    print(f"  Backward ({n1} -> {n0}): delay={backward_delay}")
    
    assert forward_delay == 1, f"Expected forward delay=1, got {forward_delay}"
    assert backward_delay == 3, f"Expected backward delay=3, got {backward_delay}"
    print(f"\n✓ PASSED: Delays correctly mapped!")

def test_multiple_fanout_nodes():
    """Test graph with multiple nodes having fanout > 1."""
    
    print(f"\n{'='*70}")
    print("TEST: Multiple Fanout Nodes")
    print(f"{'='*70}")
    
    # Graph where nodes 0, 1, and 2 all have fanout > 1
    test_adj = {
        0: [(1, 4), (2, 3)],      # fanout=2
        1: [(3, 2), (4, 5)],      # fanout=2
        2: [(3, 1), (4, 6)],      # fanout=2
        3: [(5, 1)],              # fanout=1
        4: [(5, 2)],              # fanout=1
        5: []
    }
    
    print("\nInput adjacency list:")
    for node, neighbors in test_adj.items():
        fanout = len(neighbors)
        print(f"  Node {node}: {neighbors} (fanout={fanout})")
    
    brick = LoihiGSBrick(test_adj, source=0, destination=5, name='MultiFanout')
    nodes, edges = brick._parse_input()
    proc_nodes, proc_edges = brick._preprocess_fanout(nodes, edges)
    
    # Build fanout map
    fanout_map = defaultdict(list)
    for u, v, c in proc_edges:
        fanout_map[u].append((v, c))
    
    print(f"\nAfter preprocessing:")
    nodes_with_violations = []
    for node in proc_nodes:
        edges_out = fanout_map[node]
        if len(edges_out) > 1:
            costs = [c for v, c in edges_out]
            all_one = all(c == 1 for c in costs)
            status = "✓" if all_one else "✗"
            print(f"  {status} Node {node}: fanout={len(edges_out)}, costs={costs}")
            if not all_one:
                nodes_with_violations.append(node)
    
    assert len(nodes_with_violations) == 0, f"Nodes {nodes_with_violations} violate fanout constraint"
    print(f"\n✓ PASSED: All multi-fanout nodes satisfy constraint")


def test_high_fanout():
    """Test node with very high fanout (>5 outgoing edges)."""
    
    print(f"\n{'='*70}")
    print("TEST: High Fanout Node")
    print(f"{'='*70}")
    
    # Node 0 has fanout=7
    test_adj = {
        0: [(i, i+1) for i in range(1, 8)],  # 7 edges with different costs
        **{i: [] for i in range(1, 8)}
    }
    
    print(f"\nNode 0 has fanout={len(test_adj[0])} with costs: {[c for _, c in test_adj[0]]}")
    
    brick = LoihiGSBrick(test_adj, source=0, destination=7, name='HighFanout')
    nodes, edges = brick._parse_input()
    proc_nodes, proc_edges = brick._preprocess_fanout(nodes, edges)
    
    fanout_map = defaultdict(list)
    for u, v, c in proc_edges:
        fanout_map[u].append((v, c))
    
    # Node 0 should have fanout=7 with all costs=1
    node_0_edges = fanout_map[0]
    costs = [c for v, c in node_0_edges]
    
    print(f"After preprocessing:")
    print(f"  Node 0: fanout={len(node_0_edges)}, costs={costs}")
    
    assert len(node_0_edges) == 7, f"Expected fanout=7, got {len(node_0_edges)}"
    assert all(c == 1 for c in costs), f"Not all costs are 1: {costs}"
    
    # Count auxiliary nodes created
    aux_nodes = [n for n in proc_nodes if '__aux__' in str(n)]
    print(f"  Auxiliary nodes created: {len(aux_nodes)}")
    # All 7 edges have costs > 1 (2,3,4,5,6,7,8), so all need auxiliary nodes
    assert len(aux_nodes) == 7, f"Expected 7 auxiliary nodes (all edges have cost > 1), got {len(aux_nodes)}"
    
    print(f"\n✓ PASSED: High fanout handled correctly")


def test_cost_one_edges():
    """Test that edges with cost=1 don't create auxiliary nodes."""
    
    print(f"\n{'='*70}")
    print("TEST: Cost=1 Edges (No Auxiliary Nodes Needed)")
    print(f"{'='*70}")
    
    test_adj = {
        0: [(1, 1), (2, 1), (3, 1)],  # fanout=3, all cost=1
        1: [], 2: [], 3: []
    }
    
    print(f"\nNode 0: fanout=3, all costs=1 (no aux nodes should be created)")
    
    brick = LoihiGSBrick(test_adj, source=0, destination=1, name='AllOnes')
    nodes, edges = brick._parse_input()
    proc_nodes, proc_edges = brick._preprocess_fanout(nodes, edges)
    
    aux_nodes = [n for n in proc_nodes if '__aux__' in str(n)]
    
    print(f"Auxiliary nodes created: {len(aux_nodes)}")
    print(f"Total nodes: original={len(nodes)}, processed={len(proc_nodes)}")
    
    assert len(aux_nodes) == 0, f"No auxiliary nodes should be created, but found {len(aux_nodes)}"
    assert len(proc_nodes) == len(nodes), "Node count should be unchanged"
    
    print(f"\n✓ PASSED: No unnecessary auxiliary nodes created")


def test_mixed_fanout_costs():
    """Test node with mixed costs (some 1, some >1)."""
    
    print(f"\n{'='*70}")
    print("TEST: Mixed Costs in Fanout")
    print(f"{'='*70}")
    
    test_adj = {
        0: [(1, 1), (2, 5), (3, 1), (4, 3)],  # mix of cost=1 and cost>1
        1: [], 2: [], 3: [], 4: []
    }
    
    print(f"\nNode 0: fanout=4, costs=[1, 5, 1, 3]")
    print(f"Expected: 2 auxiliary nodes (for costs 5 and 3)")
    
    brick = LoihiGSBrick(test_adj, source=0, destination=1, name='MixedCosts')
    nodes, edges = brick._parse_input()
    proc_nodes, proc_edges = brick._preprocess_fanout(nodes, edges)
    
    aux_nodes = [n for n in proc_nodes if '__aux__' in str(n)]
    
    fanout_map = defaultdict(list)
    for u, v, c in proc_edges:
        fanout_map[u].append((v, c))
    
    node_0_edges = fanout_map[0]
    costs = [c for v, c in node_0_edges]
    
    print(f"\nAfter preprocessing:")
    print(f"  Node 0: fanout={len(node_0_edges)}, costs={costs}")
    print(f"  Auxiliary nodes: {len(aux_nodes)}")
    
    assert len(aux_nodes) == 2, f"Expected 2 auxiliary nodes, got {len(aux_nodes)}"
    assert all(c == 1 for c in costs), f"All costs from node 0 should be 1: {costs}"
    
    print(f"\n✓ PASSED: Mixed costs handled correctly")


def test_chain_preservation():
    """Test that simple chains without fanout remain unchanged."""
    
    print(f"\n{'='*70}")
    print("TEST: Chain Preservation (No Fanout)")
    print(f"{'='*70}")
    
    test_adj = {
        0: [(1, 5)],
        1: [(2, 3)],
        2: [(3, 7)],
        3: []
    }
    
    print(f"\nLinear chain: 0→1→2→3 (no fanout anywhere)")
    
    brick = LoihiGSBrick(test_adj, source=0, destination=3, name='Chain')
    nodes, edges = brick._parse_input()
    proc_nodes, proc_edges = brick._preprocess_fanout(nodes, edges)
    
    aux_nodes = [n for n in proc_nodes if '__aux__' in str(n)]
    
    print(f"Auxiliary nodes created: {len(aux_nodes)}")
    assert len(aux_nodes) == 0, f"No auxiliary nodes should be created for chain, found {len(aux_nodes)}"
    assert len(proc_edges) == len(edges), "Edge count should be unchanged"
    
    # Verify costs preserved
    original_costs = {(u, v): c for u, v, c in edges}
    processed_costs = {(u, v): c for u, v, c in proc_edges}
    
    assert original_costs == processed_costs, "Costs should be preserved in chain"
    
    print(f"\n✓ PASSED: Chain remains unchanged")


def test_full_graph_construction():
    """Test that complete scaffold/graph construction works with complex fanout."""
    
    print(f"\n{'='*70}")
    print("TEST: Full Graph Construction")
    print(f"{'='*70}")
    
    test_adj = {
        0: [(1, 10), (2, 5)],     # fanout=2, need aux nodes
        1: [(3, 1)],              # fanout=1
        2: [(3, 2), (4, 3)],      # fanout=2, need aux nodes
        3: [(5, 1)],              # fanout=1
        4: [(5, 4)],              # fanout=1
        5: []
    }
    
    brick = LoihiGSBrick(test_adj, source=0, destination=5, name='FullGraph')
    scaffold = Scaffold()
    scaffold.add_brick(brick, output=True)
    scaffold.lay_bricks()
    graph = scaffold.graph
    
    print(f"\nGraph constructed:")
    print(f"  Total neurons: {len(graph.nodes)}")
    print(f"  Total synapses: {len(graph.edges)}")
    
    # Verify all edges have correct direction and delays
    forward_count = 0
    backward_count = 0
    
    for u, v, data in graph.edges(data=True):
        direction = data.get('direction')
        delay = data.get('delay')
        
        assert direction in ['forward', 'backward'], f"Invalid direction: {direction}"
        assert delay >= 1, f"Delay must be >= 1, got {delay}"
        
        if direction == 'forward':
            assert delay == 1, f"Forward delay must be 1, got {delay}"
            forward_count += 1
        else:
            assert delay >= 1, f"Backward delay must be >= 1, got {delay}"
            backward_count += 1
    
    print(f"  Forward edges: {forward_count}")
    print(f"  Backward edges: {backward_count}")
    
    assert forward_count == backward_count, "Should have equal forward and backward edges"
    
    # Verify source and destination marked
    src_neuron = graph.graph['loihi_gs']['source_neuron']
    dst_neuron = graph.graph['loihi_gs']['destination_neuron']
    
    assert graph.nodes[src_neuron].get('is_source') == True
    assert graph.nodes[dst_neuron].get('is_destination') == True
    
    print(f"\n✓ PASSED: Full graph construction valid")


def test_large_costs():
    """Test edges with costs at upper bound (2^6 = 64)."""
    
    print(f"\n{'='*70}")
    print("TEST: Large Cost Values")
    print(f"{'='*70}")
    
    test_adj = {
        0: [(1, 64), (2, 32)],  # maximum cost value
        1: [(3, 63)],
        2: [(3, 1)],
        3: []
    }
    
    print(f"\nNode 0: costs=[64, 32] (at/near max cost 2^6)")
    
    brick = LoihiGSBrick(test_adj, source=0, destination=3, name='LargeCosts')
    scaffold = Scaffold()
    scaffold.add_brick(brick, output=True)
    scaffold.lay_bricks()
    graph = scaffold.graph
    
    # Check that delays were set correctly
    max_delay = 0
    for u, v, data in graph.edges(data=True):
        if data.get('direction') == 'backward':
            delay = data.get('delay')
            max_delay = max(max_delay, delay)
    
    print(f"Maximum backward delay in graph: {max_delay}")
    # Cost 64 edge is split into 1 + 63, so max delay is 63
    assert max_delay == 63, f"Expected max delay=63, got {max_delay}"
    
    print(f"\n✓ PASSED: Large costs handled correctly")


if __name__ == "__main__":
    # Run tests directly when script is executed
    test_fanout_preprocessing()
    test_delay_mapping()
    test_multiple_fanout_nodes()
    test_high_fanout()
    test_cost_one_edges()
    test_mixed_fanout_costs()
    test_chain_preservation()
    test_full_graph_construction()
    test_large_costs()
    
    print(f"\n{'='*70}")
    print(f"{'🎉 ALL TESTS PASSED! 🎉':^70}")
    print(f"{'='*70}")
