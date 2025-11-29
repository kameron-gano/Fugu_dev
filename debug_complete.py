"""Debug the complete K5 graph issue."""

from loihi_graph_search import LoihiGraphSearch, preprocess_fanout_constraint, dijkstra

# Complete K5
adj = {}
for i in range(5):
    adj[i] = [(j, abs(i-j)+1) for j in range(5) if j != i]

print("Original graph:")
for node, neighbors in adj.items():
    print(f"  {node}: {neighbors}")

# Preprocess
adj_processed = preprocess_fanout_constraint(adj)
print(f"\nPreprocessed graph ({len(adj_processed)} nodes):")
for node in sorted(adj_processed.keys()):
    print(f"  {node}: {adj_processed[node]}")

# Run Dijkstra
source, dest = 0, 4
path_dijk, cost_dijk = dijkstra(adj_processed, source, dest)
print(f"\nDijkstra: path={path_dijk}, cost={cost_dijk}")

# Run Loihi
loihi = LoihiGraphSearch(adj_processed, source, dest)
path_loihi, hops, steps = loihi.run()
print(f"Loihi: path={path_loihi}, hops={hops}, steps={steps}")

#Print forward weights after algorithm completes
print(f"\nForward weights (after pruning):")
for i in range(min(10, loihi.n_nodes)):
    node_i = loihi.idx_to_node[i]
    for j in range(min(10, loihi.n_nodes)):
        if loihi.w_forward[i][j] > 0:
            node_j = loihi.idx_to_node[j]
            print(f"  {node_i} -> {node_j}: {loihi.w_forward[i][j]}")
