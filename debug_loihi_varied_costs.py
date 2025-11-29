#!/usr/bin/env python3
"""Debug script to trace Loihi execution on test_varied_costs graph."""

import numpy as np
from loihi_graph_search import LoihiGraphSearch, preprocess_fanout_constraint, dijkstra

# Original graph
adj_orig = {
    0: [(1, 64), (2, 1)],
    1: [(3, 1)],
    2: [(3, 63)],
    3: []
}

print("=" * 70)
print("ORIGINAL GRAPH")
print("=" * 70)
for node in sorted(adj_orig.keys()):
    print(f"  {node}: {adj_orig[node]}")
print()
print("Expected paths:")
print("  0->2->3: 1 + 63 = 64 (optimal)")
print("  0->1->3: 64 + 1 = 65")
print()

# Preprocess
adj_proc = preprocess_fanout_constraint(adj_orig)
print("=" * 70)
print("PREPROCESSED GRAPH")
print("=" * 70)
for node in sorted(adj_proc.keys()):
    print(f"  {node}: {adj_proc[node]}")
print()
print("Expected paths after +1 to all edges:")
print("  0->2->3: 2 + 64 = 66 (optimal)")
print("  0->4->1->3: 1 + 64 + 2 = 67")
print()

# Run Dijkstra on preprocessed
path_dijk, cost_dijk = dijkstra(adj_proc, 0, 3)
print(f"Dijkstra on preprocessed: path={path_dijk}, cost={cost_dijk}")
print()

# Run Loihi
print("=" * 70)
print("LOIHI EXECUTION TRACE")
print("=" * 70)

loihi = LoihiGraphSearch(adj_proc, 0, 3)
dst_idx = loihi.node_to_idx[3]
loihi.s[dst_idx] = 1
loihi.spike_time[dst_idx] = 0
loihi.spike_history = [loihi.s.copy()]

print(f"t=0: Node 3 (destination) spikes")
print(f"  spike_time: {loihi.spike_time}")
print()

# Trace first 70 timesteps
for t in range(1, 71):
    v_before = loihi.v.copy()
    loihi.advance_wavefront(t)
    s_now = loihi.s.copy()
    
    # Show spikes
    newly_spiked = np.where((s_now == 1) & (loihi.spike_time == t))[0]
    if len(newly_spiked) > 0:
        for idx in newly_spiked:
            node = loihi.idx_to_node[idx]
            print(f"t={t}: Node {node} spikes (v was {v_before[idx]:.2f})")
            print(f"  spike_time: {loihi.spike_time}")
            
            # Show backward weights from this node
            back_weights = []
            for j in range(loihi.n_nodes):
                if loihi.w_backward[idx][j] > 0:
                    src_node = loihi.idx_to_node[j]
                    delay = loihi.d_backward[idx][j]
                    back_weights.append(f"{src_node}(d={delay})")
            if back_weights:
                print(f"  Backward edges: {', '.join(back_weights)}")
            print()
    
    # Stop when source spikes
    src_idx = loihi.node_to_idx[0]
    if loihi.s[src_idx] == 1:
        print(f"Source spiked at t={loihi.spike_time[src_idx]}")
        # One more step
        loihi.advance_wavefront(t + 1)
        break

print()
print("=" * 70)
print("PATH READOUT")
print("=" * 70)

# Read out path
path = [0]
current = 0
for hop in range(10):
    next_hop = loihi.read_out_next_hop(current)
    if next_hop is None or next_hop == 3:
        if next_hop == 3:
            path.append(3)
        break
    path.append(next_hop)
    current = next_hop
    print(f"Hop {hop}: {path[-2]} -> {path[-1]}")

print(f"\nFinal path: {path}")

# Map back to original nodes
path_orig = [n for n in path if n in adj_orig]
print(f"Original nodes: {path_orig}")

# Compute cost on original
if len(path_orig) > 1:
    cost = 0
    for i in range(len(path_orig) - 1):
        u, v = path_orig[i], path_orig[i + 1]
        edge_cost = next((c for dst, c in adj_orig[u] if dst == v), None)
        cost += edge_cost
        print(f"  {u}->{v}: {edge_cost}")
    print(f"Total cost: {cost}")
