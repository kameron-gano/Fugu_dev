"""Check what preprocessing does with non-uniform costs."""

from loihi_graph_search import preprocess_fanout_constraint

# Test case: node 0 has fanout with different costs
adj = {
    0: [(1, 64), (2, 1)],
    1: [(3, 1)],
    2: [(3, 63)],
    3: []
}

print("Original graph:")
for node, neighbors in adj.items():
    for neighbor, cost in neighbors:
        print(f"  {node} -> {neighbor}: cost={cost}")

adj_processed = preprocess_fanout_constraint(adj)

print(f"\nAfter preprocessing ({len(adj_processed)} nodes):")
for node in sorted(adj_processed.keys()):
    for neighbor, cost in adj_processed[node]:
        print(f"  {node} -> {neighbor}: cost={cost}")

# Trace paths
print("\nPath analysis:")
print("  Original: 0->2->3 costs 1+63=64")
print("  Original: 0->1->3 costs 64+1=65")

# After preprocessing, node 0 has fanout=2, so it gets split
if len(adj_processed) > len(adj):
    print(f"\n  After preprocessing:")
    print(f"    0->4: cost=1 (to auxiliary)")
    print(f"    4->1: cost=64")
    print(f"    Path 0->4->1->3: 1+64+1=66")
    print(f"    0->2: cost=1")
    print(f"    2->3: cost=63")
    print(f"    Path 0->2->3: 1+63=64")
    print(f"\n  PROBLEM: The cost=1 auxiliary edge adds 1 to the first path!")
    print(f"  Original cost 64 became 66, while 64 stayed 64.")
