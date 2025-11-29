"""Simpler test to understand pruning."""

from loihi_graph_search import LoihiGraphSearch, preprocess_fanout_constraint, dijkstra

# Simplest case with fanout: 0->{(1,2), (2,1)}, both go to 3
adj = {
    0: [(1, 2), (2, 1)],
    1: [(3, 1)],
    2: [(3, 2)],
    3: []
}

print("Original:")
for n, edges in adj.items():
    print(f"  {n}: {edges}")

adj_processed = preprocess_fanout_constraint(adj)
print(f"\nPreprocessed ({len(adj_processed)} nodes):")
for n in sorted(adj_processed.keys()):
    print(f"  {n}: {adj_processed[n]}")

# Expected shortest path: 0->2->3 (cost 1+2=3)
# Alternative path: 0->aux->1->3 (cost 1+2+1=4)

source, dest = 0, 3
path_dijk, cost_dijk = dijkstra(adj_processed, source, dest)
print(f"\nDijkstra: {path_dijk}, cost={cost_dijk}")

loihi = LoihiGraphSearch(adj_processed, source, dest)

# Show initial backward edges
print("\nBackward edges (initial):")
for i in range(loihi.n_nodes):
    for j in range(loihi.n_nodes):
        if loihi.w_backward[i][j] > 0:
            print(f"  {loihi.idx_to_node[j]} -> {loihi.idx_to_node[i]}: delay={loihi.d_backward[i][j]}")

# Enable debug output
import loihi_graph_search
original_advance = loihi.advance_wavefront

def debug_advance(t):
    print(f"\n=== t={t} ===")
    result = original_advance(t)
    # Show which neurons spiked
    spiking = [loihi.idx_to_node[i] for i in range(loihi.n_nodes) if loihi.spike_time[i] == t]
    if spiking:
        print(f"Neurons spiking NOW: {spiking}")
    # Show remaining backward edges
    print("Backward edges remaining:")
    for i in range(loihi.n_nodes):
        for j in range(loihi.n_nodes):
            if loihi.w_backward[i][j] > 0:
                print(f"  {loihi.idx_to_node[j]} -> {loihi.idx_to_node[i]}: delay={loihi.d_backward[i][j]}")
    return result

loihi.advance_wavefront = debug_advance

path, hops, steps = loihi.run(max_steps=20)
print(f"\nLoihi: {path}, cost calculation:")
cost = 0
for i in range(len(path)-1):
    u, v = path[i], path[i+1]
    c = next((cost for dst, cost in adj_processed[u] if dst == v), None)
    if c:
        print(f"  {u}->{v}: {c}")
        cost += c
print(f"Total: {cost}")
