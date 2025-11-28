#!/usr/bin/env python3
"""
Rigorous tests for LoihiGSBrick preprocessing logic.

The Loihi graph search algorithm requires:
1. Graph reversal: Every outgoing edge becomes incoming, and vice versa
2. Fanout constraint: Nodes with fanout > 1 must have all outgoing edges with cost 1
   - If a branching node has outgoing edges with cost c > 1, insert auxiliary nodes
   - Original edge (u -> v, cost=c) becomes (u -> aux, cost=1) + (aux -> v, cost=c-1)
"""

import pytest
import networkx as nx
from fugu.bricks.loihi_gs_brick import LoihiGSBrick
from fugu.scaffold.scaffold import Scaffold


class TestLoihiGSPreprocessing:
    """Test suite for graph preprocessing correctness."""

    def test_simple_graph_reversal(self):
        """Test that a simple directed graph is correctly reversed."""
        # Original graph: 0 -> 1 -> 2 (costs all 1)
        adj = {
            0: [(1, 1)],
            1: [(2, 1)],
            2: []
        }
        
        brick = LoihiGSBrick(adj, source=0, destination=2, name='test')
        nodes, edges = brick._parse_input()
        
        # Before reversal: edges are 0->1, 1->2
        assert (0, 1, 1) in edges
        assert (1, 2, 1) in edges
        
        # Apply reversal
        nodes, reversed_edges = brick._reverse_graph(nodes, edges)
        
        # After reversal: edges should be 1->0, 2->1 (reversed direction, same cost)
        assert (1, 0, 1) in reversed_edges
        assert (2, 1, 1) in reversed_edges
        
        # Original edges should not exist
        assert (0, 1, 1) not in reversed_edges
        assert (1, 2, 1) not in reversed_edges

    def test_fanout_single_node_multiple_edges_same_cost(self):
        """Test fanout preprocessing when a node has multiple outgoing edges with cost 1."""
        # Original: 0 -> 1 (cost 1), 0 -> 2 (cost 1)
        # After reversal: 1 -> 0, 2 -> 0 (node 0 has fanout 2)
        # Since costs are 1, no auxiliary nodes needed
        adj = {
            0: [(1, 1), (2, 1)],
            1: [],
            2: []
        }
        
        brick = LoihiGSBrick(adj, source=0, destination=2, name='test')
        nodes, edges = brick._parse_input()
        
        # Reverse the graph
        nodes, edges = brick._reverse_graph(nodes, edges)
        
        # After reversal: edges are 1->0, 2->0
        # Node 0 now has fanout 2 in the reversed graph? No, it has IN-degree 2
        # We need to think about which node has fanout...
        # Actually, in reversed graph: 1->0, 2->0 means node 1 and 2 each have fanout 1
        # So no fanout constraint violation
        
        # Apply fanout preprocessing
        proc_nodes, proc_edges = brick._preprocess_fanout(nodes, edges)
        
        # Should have exactly 3 nodes (no auxiliary nodes for cost=1 edges)
        assert len(proc_nodes) == 3, f"Expected 3 nodes, got {len(proc_nodes)}: {proc_nodes}"
        assert set(proc_nodes) == {0, 1, 2}

    def test_fanout_with_high_cost_edge(self):
        """Test fanout preprocessing with one high-cost edge requiring auxiliary node."""
        # Original: 0 -> 1 (cost 1), 0 -> 2 (cost 5)
        # After reversal: 1 -> 0 (cost 1), 2 -> 0 (cost 5)
        # In reversed graph, nodes 1 and 2 each have fanout 1 (no constraint violation)
        # But if we want to test fanout, we need a graph where after reversal, 
        # one node has multiple outgoing edges with different costs
        
        # Better example: 1 -> 0 (cost 1), 2 -> 0 (cost 1), 0 -> 3 (cost 5)
        # After reversal: 0 -> 1 (cost 1), 0 -> 2 (cost 1), 3 -> 0 (cost 5)
        # Node 0 has fanout 2 after reversal, but both edges have cost 1, so OK
        
        # Let's use: 0 -> 1 (cost 3), 0 -> 2 (cost 1)
        # After reversal: 1 -> 0 (cost 3), 2 -> 0 (cost 1)
        # Neither has fanout > 1, so let's create a chain with branch
        
        # Graph: 1 -> 0 (cost 1), 2 -> 0 (cost 5), 0 -> 3 (cost 2), 0 -> 4 (cost 3)
        # After reversal: 0 -> 1, 0 -> 2 (cost 5), 3 -> 0, 4 -> 0
        # Hmm, still no fanout issue
        
        # Let me think differently: create adjacency that AFTER reversal has fanout > 1
        # Original: 1 -> 2 (cost 5), 3 -> 2 (cost 3)
        # Reversed: 2 -> 1 (cost 5), 2 -> 3 (cost 3)
        # Now node 2 has fanout 2 with different costs!
        
        adj = {
            1: [(2, 5)],
            3: [(2, 3)],
            2: []
        }
        
        brick = LoihiGSBrick(adj, source=1, destination=2, name='test')
        nodes, edges = brick._parse_input()
        
        # Reverse
        nodes, edges = brick._reverse_graph(nodes, edges)
        # Now edges are: 2->1 (cost 5), 2->3 (cost 3)
        
        # Preprocess fanout
        proc_nodes, proc_edges = brick._preprocess_fanout(nodes, edges)
        
        # Node 2 has fanout 2 with costs 5 and 3 (both > 1)
        # Should create 2 auxiliary nodes
        aux_nodes = [n for n in proc_nodes if '__aux__' in str(n)]
        assert len(aux_nodes) == 2, f"Expected 2 auxiliary nodes, got {len(aux_nodes)}: {aux_nodes}"
        
        # All edges from node 2 should now have cost 1
        edges_from_2 = [(u, v, c) for u, v, c in proc_edges if u == 2]
        assert len(edges_from_2) == 2, f"Expected 2 edges from node 2, got {len(edges_from_2)}"
        for u, v, c in edges_from_2:
            assert c == 1, f"Branching edge {u}->{v} should have cost 1, got {c}"

    def test_fanout_all_high_cost_edges(self):
        """Test fanout with multiple high-cost edges from one node."""
        # Node 0 fans out to 3 neighbors with costs 3, 4, 2
        adj = {
            0: [(1, 3), (2, 4), (3, 2)],
            1: [], 2: [], 3: []
        }
        
        brick = LoihiGSBrick(adj, source=1, destination=0, name='test')
        nodes, edges = brick._parse_input()
        proc_nodes, proc_edges = brick._preprocess_fanout(nodes, edges)
        
        # Should create 3 auxiliary nodes (one for each high-cost edge)
        aux_nodes = [n for n in proc_nodes if '__aux__' in str(n)]
        assert len(aux_nodes) == 3, f"Expected 3 auxiliary nodes, got {len(aux_nodes)}"
        
        # All edges from node 0 should now have cost 1
        edges_from_0 = [(u, v, c) for u, v, c in proc_edges if u == 0]
        assert len(edges_from_0) == 3
        for u, v, c in edges_from_0:
            assert c == 1, f"Edge from branching node should have cost 1, got {c}"
        
        # Each auxiliary node should have exactly one outgoing edge
        for aux in aux_nodes:
            aux_out_edges = [(u, v, c) for u, v, c in proc_edges if u == aux]
            assert len(aux_out_edges) == 1, f"Auxiliary node {aux} should have 1 outgoing edge"

    def test_path_cost_preservation(self):
        """Test that total path cost is preserved after preprocessing."""
        # Original path: 0 -> 1 (cost 5) -> 2 (cost 3)
        # Total cost should be 8
        adj = {
            0: [(1, 5)],
            1: [(2, 3)],
            2: []
        }
        
        brick = LoihiGSBrick(adj, source=0, destination=2, name='test')
        nodes, edges = brick._parse_input()
        
        # Calculate original path cost
        original_cost = 0
        current = 0
        path = [0, 1, 2]
        edge_dict = {(u, v): c for u, v, c in edges}
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            original_cost += edge_dict.get((u, v), 0)
        
        # Process
        proc_nodes, proc_edges = brick._preprocess_fanout(nodes, edges)
        
        # Build edge lookup for processed graph
        proc_edge_dict = {}
        for u, v, c in proc_edges:
            proc_edge_dict[(u, v)] = c
        
        # Find path in processed graph (may include auxiliary nodes)
        # Start at 0, follow edges to reach 2
        def find_path_cost(start, end, edge_map):
            """DFS to find path and calculate cost."""
            visited = set()
            
            def dfs(node, target, cost):
                if node == target:
                    return cost
                if node in visited:
                    return None
                visited.add(node)
                
                for (u, v), c in edge_map.items():
                    if u == node and v not in visited:
                        result = dfs(v, target, cost + c)
                        if result is not None:
                            return result
                return None
            
            return dfs(start, end, 0)
        
        processed_cost = find_path_cost(0, 2, proc_edge_dict)
        assert processed_cost == original_cost, \
            f"Path cost not preserved: original={original_cost}, processed={processed_cost}"

    def test_complex_graph_with_multiple_branching_nodes(self):
        """Test preprocessing on a complex graph with multiple fanout nodes."""
        # Graph with two branching nodes
        adj = {
            0: [(1, 2), (2, 3)],  # fanout 2, mixed costs
            1: [(3, 1)],
            2: [(3, 4), (4, 2)],  # fanout 2, mixed costs
            3: [(5, 1)],
            4: [(5, 3)],
            5: []
        }
        
        brick = LoihiGSBrick(adj, source=0, destination=5, name='test')
        nodes, edges = brick._parse_input()
        proc_nodes, proc_edges = brick._preprocess_fanout(nodes, edges)
        
        # Identify branching nodes in original graph
        out_degree = {}
        for u, v, c in edges:
            out_degree[u] = out_degree.get(u, 0) + 1
        
        branching_nodes = {u for u, deg in out_degree.items() if deg > 1}
        
        # For each branching node, verify all outgoing edges have cost 1
        for node in branching_nodes:
            out_edges = [(u, v, c) for u, v, c in proc_edges if u == node]
            for u, v, c in out_edges:
                assert c == 1, f"Branching node {node} edge to {v} should have cost 1, got {c}"

    def test_no_preprocessing_needed(self):
        """Test graph that needs no preprocessing (all costs 1, no high fanout)."""
        adj = {
            0: [(1, 1)],
            1: [(2, 1)],
            2: [(3, 1)],
            3: []
        }
        
        brick = LoihiGSBrick(adj, source=0, destination=3, name='test')
        nodes, edges = brick._parse_input()
        proc_nodes, proc_edges = brick._preprocess_fanout(nodes, edges)
        
        # No auxiliary nodes should be created
        assert len(proc_nodes) == len(nodes), "No auxiliary nodes should be added"
        assert len(proc_edges) == len(edges), "Edge count should not change"

    def test_auxiliary_node_single_fanout(self):
        """Verify auxiliary nodes themselves have fanout = 1."""
        adj = {
            0: [(1, 5), (2, 3)],
            1: [], 2: []
        }
        
        brick = LoihiGSBrick(adj, source=1, destination=0, name='test')
        nodes, edges = brick._parse_input()
        proc_nodes, proc_edges = brick._preprocess_fanout(nodes, edges)
        
        # Count outdegree for all nodes
        out_degree = {}
        for u, v, c in proc_edges:
            out_degree[u] = out_degree.get(u, 0) + 1
        
        # Find auxiliary nodes
        aux_nodes = [n for n in proc_nodes if '__aux__' in str(n)]
        
        # All auxiliary nodes must have fanout exactly 1
        for aux in aux_nodes:
            assert out_degree.get(aux, 0) == 1, \
                f"Auxiliary node {aux} should have fanout 1, got {out_degree.get(aux, 0)}"

    def test_edge_reversal_in_neural_graph(self):
        """Test that the actual neural graph has reversed edges with correct directions."""
        # Original: 0 -> 1 -> 2
        adj = {
            0: [(1, 2)],
            1: [(2, 3)],
            2: []
        }
        
        brick = LoihiGSBrick(adj, source=0, destination=2, name='test')
        scaffold = Scaffold()
        scaffold.add_brick(brick)
        scaffold.lay_bricks()
        
        graph = scaffold.graph
        
        # Collect forward and backward edges
        forward_edges = []
        backward_edges = []
        
        for u, v, data in graph.edges(data=True):
            if data.get('direction') == 'forward':
                forward_edges.append((u, v, data.get('delay')))
            elif data.get('direction') == 'backward':
                backward_edges.append((u, v, data.get('delay')))
        
        # Forward edges should have delay 1 (for readout)
        for u, v, delay in forward_edges:
            assert delay == 1, f"Forward edge {u}->{v} should have delay 1, got {delay}"
        
        # Backward edges should have delay equal to original cost
        # The graph should be reversed, so backward edges go opposite to forward
        # Each forward edge should have a corresponding backward edge
        assert len(forward_edges) > 0, "Should have forward edges"
        assert len(backward_edges) > 0, "Should have backward edges"
        
        # For each forward edge (u->v), there should be a backward edge (v->u)
        forward_pairs = {(u, v) for u, v, _ in forward_edges}
        backward_pairs = {(u, v) for u, v, _ in backward_edges}
        
        for u, v in forward_pairs:
            assert (v, u) in backward_pairs, \
                f"Forward edge {u}->{v} missing corresponding backward edge {v}->{u}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
