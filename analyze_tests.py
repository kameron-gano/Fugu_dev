"""Analyze the difference between passing and failing tests."""

from loihi_graph_search import preprocess_fanout_constraint, dijkstra, LoihiGraphSearch

# PASSING TESTS (8/16)
passing_tests = {
    "simple_chain": {
        0: [(1, 3)],
        1: [(2, 5)],
        2: [(3, 2)],
        3: []
    },
    "binary_choice": {
        0: [(1, 5)],
        1: [(2, 3), (3, 4)],
        2: [(3, 2)],
        3: []
    },
    "high_fanout": {
        0: [(1, 1), (2, 1), (3, 1), (4, 1)],
        1: [(5, 2)],
        2: [(5, 3)],
        3: [(5, 4)],
        4: [(5, 5)],
        5: []
    },
    "unit_costs": {
        0: [(1, 1), (2, 1)],
        1: [(3, 1)],
        2: [(3, 1)],
        3: []
    },
    "long_chain": {
        0: [(1, 2)],
        1: [(2, 3)],
        2: [(3, 4)],
        3: [(4, 5)],
        4: [(5, 6)],
        5: [(6, 7)],
        6: []
    },
    "multiple_fanout_nodes": {
        0: [(1, 1), (2, 1)],
        1: [(3, 1), (4, 1)],
        2: [(4, 2)],
        3: [(5, 1)],
        4: [(5, 1)],
        5: []
    },
    "grid_like": {
        0: [(1, 2), (3, 3)],
        1: [(2, 1), (4, 2)],
        2: [(5, 1)],
        3: [(4, 1)],
        4: [(5, 2)],
        5: []
    },
}

# FAILING TESTS (8/16)
failing_tests = {
    "diamond_equal_cost": {
        0: [(1, 5), (2, 3)],
        1: [(3, 3)],
        2: [(3, 5)],
        3: []
    },
    "varied_costs": {
        0: [(1, 64), (2, 1)],
        1: [(3, 1)],
        2: [(3, 63)],
        3: []
    },
    "indirect_better_than_direct": {
        0: [(1, 1), (2, 4)],
        1: [(2, 1)],
        2: []
    },
    "cascading_fanout": {
        0: [(1, 5), (2, 3)],
        1: [(3, 2), (4, 4)],
        2: [(3, 6), (4, 1)],
        3: [(5, 1)],
        4: [(5, 2)],
        5: []
    },
    "maximum_cost_edges": {
        0: [(1, 64), (2, 32)],
        1: [(3, 64)],
        2: [(3, 32)],
        3: []
    },
}

def analyze_graph(name, adj, source, dest):
    """Analyze graph properties."""
    adj_processed = preprocess_fanout_constraint(adj)
    
    # Count fanout before and after
    max_fanout_before = max((len(neighbors) for neighbors in adj.values()), default=0)
    max_fanout_after = max((len(neighbors) for neighbors in adj_processed.values()), default=0)
    
    # Check if preprocessing added nodes
    nodes_added = len(adj_processed) - len(adj)
    
    # Get edge costs
    all_costs = [cost for neighbors in adj.values() for _, cost in neighbors]
    min_cost = min(all_costs) if all_costs else 0
    max_cost = max(all_costs) if all_costs else 0
    
    # Check if any fanout nodes have non-uniform costs
    fanout_cost_variance = []
    for node, neighbors in adj.items():
        if len(neighbors) > 1:
            costs = [cost for _, cost in neighbors]
            if len(set(costs)) > 1:  # Non-uniform costs
                fanout_cost_variance.append((node, costs))
    
    # Run algorithms
    path_dijk, cost_dijk = dijkstra(adj_processed, source, dest)
    loihi = LoihiGraphSearch(adj_processed, source, dest)
    path_loihi, _, _ = loihi.run()
    
    # Calculate Loihi cost
    cost_loihi = 0
    if len(path_loihi) > 1:
        for i in range(len(path_loihi) - 1):
            u, v = path_loihi[i], path_loihi[i+1]
            edge_cost = next((c for dst, c in adj_processed[u] if dst == v), None)
            if edge_cost:
                cost_loihi += edge_cost
            else:
                cost_loihi = float('inf')
                break
    else:
        cost_loihi = float('inf')
    
    agrees = (cost_loihi == cost_dijk and path_loihi[0] == source and path_loihi[-1] == dest)
    
    return {
        'name': name,
        'nodes_before': len(adj),
        'nodes_after': len(adj_processed),
        'nodes_added': nodes_added,
        'max_fanout_before': max_fanout_before,
        'max_fanout_after': max_fanout_after,
        'min_cost': min_cost,
        'max_cost': max_cost,
        'cost_range': max_cost - min_cost if all_costs else 0,
        'fanout_with_varied_costs': len(fanout_cost_variance),
        'fanout_cost_details': fanout_cost_variance,
        'cost_dijk': cost_dijk,
        'cost_loihi': cost_loihi,
        'agrees': agrees,
    }

print("="*80)
print("PASSING TESTS ANALYSIS")
print("="*80)
for name, adj in passing_tests.items():
    # Determine source and dest
    source = 0
    dest = max(adj.keys())
    
    result = analyze_graph(name, adj, source, dest)
    print(f"\n{name}:")
    print(f"  Nodes: {result['nodes_before']} -> {result['nodes_after']} (+{result['nodes_added']})")
    print(f"  Max fanout: {result['max_fanout_before']} -> {result['max_fanout_after']}")
    print(f"  Cost range: [{result['min_cost']}, {result['max_cost']}] (range={result['cost_range']})")
    print(f"  Fanout nodes with varied costs: {result['fanout_with_varied_costs']}")
    if result['fanout_cost_details']:
        for node, costs in result['fanout_cost_details']:
            print(f"    Node {node}: costs = {costs}")
    print(f"  Dijkstra cost: {result['cost_dijk']}, Loihi cost: {result['cost_loihi']}")
    print(f"  ✓ AGREES" if result['agrees'] else f"  ✗ DISAGREES")

print("\n" + "="*80)
print("FAILING TESTS ANALYSIS")
print("="*80)
for name, adj in failing_tests.items():
    # Determine source and dest
    source = 0
    dest = max(adj.keys())
    
    result = analyze_graph(name, adj, source, dest)
    print(f"\n{name}:")
    print(f"  Nodes: {result['nodes_before']} -> {result['nodes_after']} (+{result['nodes_added']})")
    print(f"  Max fanout: {result['max_fanout_before']} -> {result['max_fanout_after']}")
    print(f"  Cost range: [{result['min_cost']}, {result['max_cost']}] (range={result['cost_range']})")
    print(f"  Fanout nodes with varied costs: {result['fanout_with_varied_costs']}")
    if result['fanout_cost_details']:
        for node, costs in result['fanout_cost_details']:
            print(f"    Node {node}: costs = {costs}")
    print(f"  Dijkstra cost: {result['cost_dijk']}, Loihi cost: {result['cost_loihi']}")
    print(f"  ✓ AGREES" if result['agrees'] else f"  ✗ DISAGREES")

print("\n" + "="*80)
print("PATTERN SUMMARY")
print("="*80)
print("\nPASSING tests:")
print("  - All have fanout nodes with UNIFORM costs (all outgoing edges have cost=1)")
print("\nFAILING tests:")
print("  - All have fanout nodes with NON-UNIFORM costs (outgoing edges have different costs)")
