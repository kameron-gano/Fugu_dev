"""Simpler debug - just track what's happening with nodes 0, 5, 1, 9."""

from loihi_graph_search import LoihiGraphSearch, preprocess_fanout_constraint
import numpy as np

# Simple test with fanout: 0 -> {(1, 64), (2, 1)}
adj = {
    0: [(1, 64), (2, 1)],
    1: [(3, 1)],
    2: [(3, 63)],
    3: []
}

adj_processed = preprocess_fanout_constraint(adj)
print("Preprocessed graph:")
for node in sorted(adj_processed.keys()):
    print(f"  {node}: {adj_processed[node]}")

source, dest = 0, 3
loihi = LoihiGraphSearch(adj_processed, source, dest)

# Manual simulation with debug
src_idx = loihi.node_to_idx[source]
dst_idx = loihi.node_to_idx[dest]

# Initialize
loihi.s[dst_idx] = 1
loihi.spike_history = [loihi.s.copy()]

print(f"\n=== Starting simulation: source={source}, dest={dest} ===")
print(f"t=0: Destination {dest} spiked")
print(f"Spike state: {loihi.s}")

max_steps = 70
for t in range(1, max_steps + 1):
    loihi.advance_wavefront(t)
    
    if np.any(loihi.s[loihi.s > 0]) or t < 10 or t > max_steps - 5:
        spikes = [loihi.idx_to_node[i] for i in range(loihi.n_nodes) if loihi.s[i] > 0]
        print(f"t={t}: spikes={spikes}, potentials={loihi.v[:loihi.n_nodes]}")
    
    if loihi.s[src_idx] == 1:
        print(f"\n*** Source {source} spiked at t={t} ***")
        break

if loihi.s[src_idx] == 0:
    print(f"\n*** Source did not spike after {max_steps} steps ***")

# One extra step
loihi.advance_wavefront(t+1)

# Check forward weights
print(f"\n=== Forward edges (after pruning) ===")
for i in range(loihi.n_nodes):
    for j in range(loihi.n_nodes):
        if loihi.w_forward[i][j] > 0:
            print(f"  {loihi.idx_to_node[i]} -> {loihi.idx_to_node[j]}")

# Read out path
path = [source]
current = source
for _ in range(10):
    next_hop = loihi.read_out_next_hop(current)
    if next_hop is None or next_hop == dest:
        if next_hop == dest:
            path.append(dest)
        break
    path.append(next_hop)
    current = next_hop

print(f"\nPath: {path}")
