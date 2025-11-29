"""Debug script to understand why Loihi algorithm returns empty paths."""

import numpy as np
from loihi_graph_search import LoihiGraphSearch, preprocess_fanout_constraint

# Simple test: 0 -> 1 -> 2 -> 3
adj = {
    0: [(1, 3)],
    1: [(2, 5)],
    2: [(3, 2)],
    3: []
}

print("Original graph:")
for node, neighbors in adj.items():
    print(f"  {node}: {neighbors}")

# Preprocess
adj_processed = preprocess_fanout_constraint(adj)
print("\nAfter preprocessing:")
for node, neighbors in adj_processed.items():
    print(f"  {node}: {neighbors}")

# Create Loihi instance
source, dest = 0, 3
loihi = LoihiGraphSearch(adj_processed, source, dest)

print(f"\n{'='*70}")
print(f"Running Loihi: source={source}, dest={dest}")
print(f"{'='*70}")

# Print graph structure
print(f"\nBackward weights (should propagate from dest={dest} backwards):")
for i in range(loihi.n_nodes):
    for j in range(loihi.n_nodes):
        if loihi.w_backward[i][j] > 0:
            delay = loihi.d_backward[i][j]
            node_from = loihi.idx_to_node[j]
            node_to = loihi.idx_to_node[i]
            print(f"  {node_from} -> {node_to}: weight={loihi.w_backward[i][j]}, delay={delay}")

# Manually simulate to see what happens
src_idx = loihi.node_to_idx[source]
dst_idx = loihi.node_to_idx[dest]

# Initialize as run() does
loihi.s[dst_idx] = 1
loihi.spike_history = [loihi.s.copy()]

print(f"\nInitial state (t=0 - destination spiked):")
print(f"  Potentials: {loihi.v}")
print(f"  Spikes: {loihi.s}")
print(f"  Spike history: {[list(s) for s in loihi.spike_history]}")

# Run a few timesteps manually
max_steps = 20
for t in range(1, max_steps + 1):
    loihi.advance_wavefront(t)
    
    print(f"\nTimestep t={t}:")
    print(f"  Potentials: {loihi.v}")
    print(f"  Spikes: {loihi.s}")
    if np.any(loihi.s > 0):
        spiking_nodes = [loihi.idx_to_node[i] for i in range(loihi.n_nodes) if loihi.s[i] > 0]
        print(f"  Spiking nodes: {spiking_nodes}")
    
    # Check if source spiked
    if loihi.s[src_idx] == 1:
        print(f"\n*** SOURCE SPIKED at t={t} ***")
        break

if loihi.s[src_idx] == 0:
    print(f"\n*** SOURCE DID NOT SPIKE after {max_steps} steps ***")
    print(f"Source potential: {loihi.v[src_idx]}")

# Try to read out path anyway
print(f"\n{'='*70}")
print("Attempting path readout...")
print(f"{'='*70}")

path = loihi.read_out_next_hop()
print(f"Path: {path}")

# Check forward weights
print(f"\nForward weights (should show path):")
for i in range(loihi.n_nodes):
    for j in range(loihi.n_nodes):
        if loihi.w_forward[i][j] > 0:
            print(f"  {loihi.idx_to_node[i]} -> {loihi.idx_to_node[j]}: weight={loihi.w_forward[i][j]}")
