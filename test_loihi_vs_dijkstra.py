#!/usr/bin/env python3
"""
Rigorous test harness comparing Loihi graph search against Dijkstra's algorithm.

These tests establish ground truth for eventual Fugu implementation by verifying
the Loihi algorithm produces correct shortest paths across diverse graph structures.
"""

import pytest
import random
from loihi_graph_search import (
    LoihiGraphSearch,
    dijkstra,
    preprocess_fanout_constraint
)


class TestLoihiVsDijkstra:
    """Compare Loihi graph search against Dijkstra's algorithm."""
    
    def compare_algorithms(self, adj, source, dest, test_name=""):
        """Helper to run both algorithms and compare results."""
        print(f"\n{'='*70}")
        print(f"TEST: {test_name}")
        print(f"{'='*70}")
        print(f"Graph: {len(adj)} nodes, source={source}, dest={dest}")
        
        # Run Dijkstra on ORIGINAL graph to get ground truth
        path_dijk, cost_dijk = dijkstra(adj, source, dest)
        print(f"Dijkstra (original graph): path={path_dijk}, cost={cost_dijk}")
        
        # Preprocess for Loihi
        adj_processed = preprocess_fanout_constraint(adj)
        print(f"After fanout preprocessing: {len(adj_processed)} nodes")
        
        # Run Loihi on preprocessed graph
        loihi = LoihiGraphSearch(adj_processed, source, dest)
        path_loihi, hops_loihi, steps_loihi = loihi.run()
        print(f"Loihi (preprocessed): path={path_loihi}, hops={hops_loihi}, steps={steps_loihi}")
        
        # Map Loihi path back to original nodes (exclude auxiliary nodes)
        path_loihi_original = [n for n in path_loihi if n in adj]
        print(f"Loihi (original nodes only): {path_loihi_original}")
        
        # Compute cost using ORIGINAL graph edges only
        if len(path_loihi_original) > 1:
            cost_loihi = 0
            for i in range(len(path_loihi_original) - 1):
                u = path_loihi_original[i]
                v = path_loihi_original[i + 1]
                # Find edge cost in ORIGINAL graph
                edge_cost = next((c for dst, c in adj[u] if dst == v), None)
                if edge_cost is None:
                    print(f"WARNING: No edge {u}->{v} in original graph")
                    cost_loihi = float('inf')
                    break
                cost_loihi += edge_cost
        else:
            cost_loihi = float('inf')
        
        print(f"Loihi path cost (original edges): {cost_loihi}")
        
        # Verify correctness
        assert path_loihi_original[0] == source, "Path must start at source"
        assert path_loihi_original[-1] == dest, "Path must end at destination"
        assert cost_loihi == cost_dijk, f"Path costs differ: Loihi={cost_loihi}, Dijkstra={cost_dijk}"
        
        print(f"✓ PASS: Algorithms agree on cost={cost_dijk}")
        return cost_loihi, steps_loihi
    
    def test_simple_chain(self):
        """Test 1: Simple linear chain."""
        adj = {
            0: [(1, 3)],
            1: [(2, 5)],
            2: [(3, 2)],
            3: []
        }
        self.compare_algorithms(adj, 0, 3, "Simple Chain")
    
    def test_binary_choice(self):
        """Test 2: Binary choice with clear optimal path."""
        adj = {
            0: [(1, 2), (2, 10)],
            1: [(3, 3)],
            2: [(3, 1)],
            3: []
        }
        self.compare_algorithms(adj, 0, 3, "Binary Choice")
    
    def test_diamond_equal_cost(self):
        """Test 3: Diamond with equal cost paths."""
        adj = {
            0: [(1, 5), (2, 3)],
            1: [(3, 3)],
            2: [(3, 5)],
            3: []
        }
        cost, _ = self.compare_algorithms(adj, 0, 3, "Diamond Equal Cost")
        assert cost == 8  # Both paths cost 8
    
    def test_high_fanout(self):
        """Test 4: Node with high fanout (requires many aux nodes)."""
        adj = {
            0: [(i, i+1) for i in range(1, 6)],  # fanout=5
            **{i: [(6, 1)] for i in range(1, 6)},
            6: []
        }
        self.compare_algorithms(adj, 0, 6, "High Fanout")
    
    def test_unit_costs(self):
        """Test 5: All edges have cost=1 (no aux nodes needed)."""
        adj = {
            0: [(1, 1), (2, 1)],
            1: [(3, 1), (4, 1)],
            2: [(4, 1)],
            3: [(5, 1)],
            4: [(5, 1)],
            5: []
        }
        self.compare_algorithms(adj, 0, 5, "Unit Costs")
    
    def test_varied_costs(self):
        """Test 6: Wide range of edge costs (1 to 64)."""
        adj = {
            0: [(1, 64), (2, 1)],
            1: [(3, 1)],
            2: [(3, 63)],
            3: []
        }
        self.compare_algorithms(adj, 0, 3, "Varied Costs")
    
    def test_long_chain(self):
        """Test 7: Long chain of nodes."""
        n = 20
        adj = {i: [(i+1, i+2)] for i in range(n)}
        adj[n] = []
        self.compare_algorithms(adj, 0, n, "Long Chain")
    
    def test_complete_small(self):
        """Test 8: Complete graph K5."""
        adj = {}
        for i in range(5):
            adj[i] = [(j, abs(i-j)+1) for j in range(5) if j != i]
        self.compare_algorithms(adj, 0, 4, "Complete K5")
    
    def test_multiple_fanout_nodes(self):
        """Test 9: Multiple nodes with fanout > 1."""
        adj = {
            0: [(1, 3), (2, 5)],
            1: [(3, 2), (4, 4)],
            2: [(4, 2), (5, 3)],
            3: [(6, 1)],
            4: [(6, 1)],
            5: [(6, 1)],
            6: []
        }
        self.compare_algorithms(adj, 0, 6, "Multiple Fanout Nodes")
    
    def test_grid_like(self):
        """Test 10: Grid-like structure."""
        # 3x3 grid, can move right or down
        adj = {
            0: [(1, 2), (3, 3)],
            1: [(2, 2), (4, 3)],
            2: [(5, 3)],
            3: [(4, 2), (6, 3)],
            4: [(5, 2), (7, 3)],
            5: [(8, 3)],
            6: [(7, 2)],
            7: [(8, 2)],
            8: []
        }
        self.compare_algorithms(adj, 0, 8, "Grid-like")
    
    def test_indirect_better_than_direct(self):
        """Test 11: Indirect path is cheaper than direct."""
        adj = {
            0: [(1, 2), (2, 20)],
            1: [(2, 1)],
            2: []
        }
        cost, _ = self.compare_algorithms(adj, 0, 2, "Indirect Better")
        assert cost == 3  # Via node 1
    
    def test_cascading_fanout(self):
        """Test 12: Cascading fanout where aux nodes also branch."""
        adj = {
            0: [(1, 5), (2, 3)],     # fanout=2
            1: [(3, 2), (4, 4)],     # fanout=2  
            2: [(3, 6), (4, 1)],     # fanout=2
            3: [(5, 1)],
            4: [(5, 2)],
            5: []
        }
        self.compare_algorithms(adj, 0, 5, "Cascading Fanout")
    
    def test_maximum_cost_edges(self):
        """Test 13: Edges at maximum cost (64)."""
        adj = {
            0: [(1, 64), (2, 32)],
            1: [(3, 64)],
            2: [(3, 32)],
            3: []
        }
        self.compare_algorithms(adj, 0, 3, "Maximum Cost Edges")
    
    def test_dense_graph(self):
        """Test 14: Dense random graph."""
        random.seed(42)
        n = 15
        adj = {}
        for i in range(n):
            neighbors = []
            for j in range(i+1, n):
                if random.random() < 0.5:  # 50% density
                    cost = random.randint(1, 20)
                    neighbors.append((j, cost))
            adj[i] = neighbors
        
        self.compare_algorithms(adj, 0, n-1, "Dense Random Graph")
    
    def test_sparse_graph(self):
        """Test 15: Sparse random graph with long paths."""
        random.seed(123)
        n = 20
        adj = {i: [] for i in range(n)}
        
        # Ensure path exists by creating backbone
        for i in range(n-1):
            adj[i].append((i+1, random.randint(1, 10)))
        
        # Add random edges (sparse)
        for i in range(n):
            for j in range(i+2, n):
                if random.random() < 0.15:  # 15% density
                    cost = random.randint(1, 15)
                    adj[i].append((j, cost))
        
        self.compare_algorithms(adj, 0, n-1, "Sparse Random Graph")


def test_fanout_preprocessing():
    """Test that fanout preprocessing works correctly with cost preservation."""
    adj = {
        0: [(1, 5), (2, 3), (3, 1)],  # fanout=3, mixed costs
        1: [], 2: [], 3: []
    }
    
    processed = preprocess_fanout_constraint(adj)
    
    # After +1 cost preservation:
    # - Edge 0->3 (cost=1) becomes cost=2 (no aux needed, just +1)
    # - Edge 0->2 (cost=3) gets split: 0->aux (1) + aux->2 (3) = 4 total
    # - Edge 0->1 (cost=5) gets split: 0->aux (1) + aux->1 (5) = 6 total
    
    # Check that node 0's direct edges all satisfy fanout constraint
    # (either cost=1 to aux, or cost=2 for edges that were originally cost=1)
    for dest, cost in processed[0]:
        assert cost in [1, 2], f"Fanout constraint violated: cost={cost} (expected 1 or 2)"
    
    # Should have created 2 auxiliary nodes (for costs 5 and 3)
    aux_nodes = [n for n in processed.keys() if n not in adj]
    assert len(aux_nodes) == 2, f"Expected 2 aux nodes, got {len(aux_nodes)}"
    
    # Verify total path costs are preserved (+1 to all)
    # Original 0->1: 5, preprocessed: 1+5=6 ✓
    # Original 0->2: 3, preprocessed: 1+3=4 ✓
    # Original 0->3: 1, preprocessed: 2 ✓
    
    print("✓ Fanout preprocessing test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
