"""Debug the varied costs test where Loihi disagrees with Dijkstra."""

from loihi_graph_search import LoihiGraphSearch, preprocess_fanout_constraint, dijkstra

# Original: 0->(1,64), 0->(2,1), 1->(3,1), 2->(3,63)
adj = {
    0: [(1, 64), (2, 1)],
    1: [(3, 1)],
    2: [(3, 63)],
    3: []
}

print("Original graph:")
for node, neighbors in adj.items():
    print(f"  {node}: {neighbors}")

# Preprocess
adj_processed = preprocess_fanout_constraint(adj)
print(f"\nPreprocessed graph ({len(adj_processed)} nodes):")
for node in sorted(adj_processed.keys()):
    print(f"  {node}: {adj_processed[node]}")

# Run Dijkstra
source, dest = 0, 3
path_dijk, cost_dijk = dijkstra(adj_processed, source, dest)
print(f"\nDijkstra: path={path_dijk}, cost={cost_dijk}")

# Run Loihi
loihi = LoihiGraphSearch(adj_processed, source, dest)

# Print backward weights BEFORE running
print(f"\nBackward edges (initial):")
for i in range(loihi.n_nodes):
    for j in range(loihi.n_nodes):
        if loihi.w_backward[i][j] > 0:
            node_i = loihi.idx_to_node[i]
            node_j = loihi.idx_to_node[j]
            delay = loihi.d_backward[i][j]
            print(f"  {node_j} -> {node_i}: delay={delay}")

path_loihi, hops, steps = loihi.run()

# Calculate actual cost
cost_loihi = 0
if len(path_loihi) > 1:
    for i in range(len(path_loihi) - 1):
        u, v = path_loihi[i], path_loihi[i+1]
        edge_cost = next((c for dst, c in adj_processed[u] if dst == v), None)
        if edge_cost:
            cost_loihi += edge_cost
            print(f"  Edge {u}->{v}: cost={edge_cost}")

print(f"\nLoihi: path={path_loihi}, cost={cost_loihi}, steps={steps}")

# Print backward weights and delays
print(f"\nBackward edges (wavefront propagation):")
for i in range(loihi.n_nodes):
    for j in range(loihi.n_nodes):
        if loihi.w_backward[i][j] > 0:
            node_i = loihi.idx_to_node[i]
            node_j = loihi.idx_to_node[j]
            delay = loihi.d_backward[i][j]
            print(f"  {node_j} -> {node_i}: delay={delay}")

# Check when each node spiked
print(f"\nSpike history ({len(loihi.spike_history)} timesteps):")
for t, spikes in enumerate(loihi.spike_history[:min(70, len(loihi.spike_history))]):
    spiking = [loihi.idx_to_node[i] for i in range(len(spikes)) if spikes[i] > 0]
    if spiking:
        print(f"  t={t}: {spiking}")
