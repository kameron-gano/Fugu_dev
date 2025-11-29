"""Detailed trace of the pruning decision."""

from loihi_graph_search import LoihiGraphSearch, preprocess_fanout_constraint, dijkstra

adj = {
    0: [(1, 2), (2, 1)],
    1: [(3, 1)],
    2: [(3, 2)],
    3: []
}

adj_processed = preprocess_fanout_constraint(adj)
print("Preprocessed graph:")
for n in sorted(adj_processed.keys()):
    print(f"  {n}: {adj_processed[n]}")
print()

source, dest = 0, 3
loihi = LoihiGraphSearch(adj_processed, source, dest)

# Manually run to t=3 when node 0 spikes
dst_idx = loihi.node_to_idx[dest]
loihi.s[dst_idx] = 1
loihi.spike_time[dst_idx] = 0
loihi.spike_history = [loihi.s.copy()]

print("=== Manual simulation ===\n")

for t in range(1, 5):
    print(f"t={t}:")
    print(f"  Before: v={loihi.v}, spike_times={loihi.spike_time}")
    
    # Manually do what advance_wavefront does
    v_prev = loihi.v.copy()
    v_new = loihi.v.copy()
    
    # Update potentials
    for i in range(loihi.n_nodes):
        input_current = 0.0
        for j in range(loihi.n_nodes):
            if loihi.w_backward[i][j] > 0:
                delay = loihi.d_backward[i][j]
                if t >= delay:
                    time_idx = t - delay
                    if time_idx < len(loihi.spike_history):
                        s_delayed = loihi.spike_history[time_idx][j]
                        if s_delayed > 0:
                            input_current += loihi.w_backward[i][j] * s_delayed
                            print(f"    Node {loihi.idx_to_node[i]} gets input from node {loihi.idx_to_node[j]} (delay={delay}, spike at t={time_idx})")
        
        v_new[i] = loihi.v[i] + input_current
    
    loihi.v = v_new
    
    # Check for threshold crossings
    newly_spiked = []
    s_new = loihi.s.copy()
    for i in range(loihi.n_nodes):
        if loihi.v[i] >= 1.0 and v_prev[i] < 1.0:
            s_new[i] = 1
            loihi.spike_time[i] = t
            newly_spiked.append(i)
            print(f"  -> Node {loihi.idx_to_node[i]} SPIKES (v crossed from {v_prev[i]:.1f} to {loihi.v[i]:.1f})")
    
    loihi.s = s_new
    
    # Prune edges
    if newly_spiked:
        print(f"  Pruning edges for newly spiked neurons: {[loihi.idx_to_node[i] for i in newly_spiked]}")
        for i in newly_spiked:
            for j in range(loihi.n_nodes):
                if loihi.w_backward[i][j] > 0:
                    delay = loihi.d_backward[i][j]
                    expected_spike_time = t - delay
                    node_i = loihi.idx_to_node[i]
                    node_j = loihi.idx_to_node[j]
                    print(f"    Check edge {node_j}->{node_i} (delay={delay}): did {node_j} spike at t={expected_spike_time}? spike_time={loihi.spike_time[j]}")
                    if loihi.spike_time[j] == expected_spike_time:
                        print(f"      YES! Pruning edge {node_j}->{node_i}")
                        loihi.w_backward[i][j] = 0.0
                        loihi.w_forward[j][i] = 0.0
    
    loihi.spike_history.append(loihi.s.copy())
    print()
    
    if t == 3:
        break

print("\n=== Result at t=3 ===")
print(f"Remaining backward edges:")
for i in range(loihi.n_nodes):
    for j in range(loihi.n_nodes):
        if loihi.w_backward[i][j] > 0:
            print(f"  {loihi.idx_to_node[j]} -> {loihi.idx_to_node[i]}: delay={loihi.d_backward[i][j]}")
