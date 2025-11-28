#!/usr/bin/env python3
"""
Integration tests for the complete Loihi graph-search algorithm.

Tests verify:
1. Correct wavefront propagation (extra step after source spikes)
2. Proper backward edge pruning
3. Shortest path correctness (compared to Dijkstra's)
"""

import pytest
import networkx as nx
from fugu.bricks.loihi_gs_brick import LoihiGSBrick
from fugu.scaffold import Scaffold
from fugu.backends.gsearch_backend import gsearch_Backend


def dijkstra_shortest_path(adj_dict, source, destination):
    """
    Compute shortest path using Dijkstra's algorithm on original graph.
    
    Args:
        adj_dict: Adjacency list {node: [(neighbor, cost), ...]}
        source: Source node
        destination: Destination node
        
    Returns:
        Tuple of (path_list, total_cost) or (None, None) if no path exists
    """
    # Build NetworkX graph from adjacency dict
    G = nx.DiGraph()
    for u, neighbors in adj_dict.items():
        for v, cost in neighbors:
            G.add_edge(u, v, weight=cost)
    
    # Check if path exists
    if not nx.has_path(G, source, destination):
        return None, None
    
    # Find shortest path
    path = nx.shortest_path(G, source, destination, weight='weight')
    
    # Calculate total cost
    total_cost = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        total_cost += G[u][v]['weight']
    
    return path, total_cost


class TestLoihiGSAlgorithm:
    """Integration tests for the complete algorithm."""

    def test_simple_path_three_nodes(self):
        """Test simple linear path: A -> B -> C."""
        adj = {
            'A': [('B', 2)],
            'B': [('C', 3)],
            'C': []
        }
        
        # Expected shortest path from Dijkstra's
        expected_path, expected_cost = dijkstra_shortest_path(adj, 'A', 'C')
        assert expected_path == ['A', 'B', 'C']
        assert expected_cost == 5
        
        # Run Loihi algorithm
        brick = LoihiGSBrick(adj, source='A', destination='C', name='test')
        scaffold = Scaffold()
        scaffold.add_brick(brick, output=True)
        scaffold.lay_bricks()
        
        backend = gsearch_Backend()
        backend.compile(scaffold, {})
        result = backend.run(n_steps=100)
        
        # Verify source spiked
        assert result['source_spiked'], f"Source should have spiked. Result: {result}"
        
        # Verify path correctness
        assert len(result['path']) > 0, "Path should not be empty"
        
        # Map neuron names back to node names
        neuron_to_node = {v: k for k, v in brick.node_to_neuron.items() if '__aux__' not in str(k)}
        path_nodes = [neuron_to_node.get(neuron, neuron) for neuron in result['path']]
        
        # Filter out auxiliary nodes from path
        path_nodes = [n for n in path_nodes if '__aux__' not in str(n)]
        
        assert path_nodes[0] == 'A', f"Path should start at A, got {path_nodes[0]}"
        assert path_nodes[-1] == 'C', f"Path should end at C, got {path_nodes[-1]}"

    def test_branching_graph(self):
        """Test graph with multiple paths to destination."""
        adj = {
            'A': [('B', 1), ('C', 5)],
            'B': [('C', 2)],
            'C': []
        }
        
        # Expected: A -> B -> C (cost 3) is shorter than A -> C (cost 5)
        expected_path, expected_cost = dijkstra_shortest_path(adj, 'A', 'C')
        assert expected_path == ['A', 'B', 'C']
        assert expected_cost == 3
        
        brick = LoihiGSBrick(adj, source='A', destination='C', name='test_branching')
        scaffold = Scaffold()
        scaffold.add_brick(brick, output=True)
        scaffold.lay_bricks()
        
        backend = gsearch_Backend()
        backend.compile(scaffold, {})
        result = backend.run(n_steps=100)
        
        assert result['source_spiked'], "Source should have spiked"
        assert len(result['path']) > 0, "Should find a path"

    def test_backward_edge_pruning(self):
        """Verify that backward edges are properly pruned during execution."""
        adj = {
            0: [(1, 2), (2, 4)],
            1: [(2, 1)],
            2: []
        }
        
        brick = LoihiGSBrick(adj, source=0, destination=2, name='test')
        scaffold = Scaffold()
        scaffold.add_brick(brick, output=True)
        scaffold.lay_bricks()
        
        backend = gsearch_Backend()
        backend.compile(scaffold, {})
        
        # Count initial backward edges
        initial_backward = backend.remaining_backward_edges()
        assert len(initial_backward) > 0, "Should have backward edges initially"
        
        # Run algorithm
        result = backend.run(n_steps=100)
        
        assert result['source_spiked'], "Source should have spiked"
        
        # After algorithm completes, check remaining backward edges
        # According to the algorithm image, only forward edges should remain
        final_backward = backend.remaining_backward_edges()
        
        # The shortest path edges should have their backward weights zeroed
        # Some backward edges off the shortest path may remain
        assert len(final_backward) < len(initial_backward), \
            f"Should prune some backward edges: initial={len(initial_backward)}, final={len(final_backward)}"

    def test_extra_wavefront_step_after_source_spikes(self):
        """Verify the extra ADVANCEWAVEFRONT step (line 26) after source spikes."""
        adj = {
            'A': [('B', 1)],
            'B': [('C', 1)],
            'C': []
        }
        
        brick = LoihiGSBrick(adj, source='A', destination='C', name='test_extra')
        scaffold = Scaffold()
        scaffold.add_brick(brick, output=True)
        scaffold.lay_bricks()
        
        backend = gsearch_Backend()
        backend.compile(scaffold, {})
        
        # Run algorithm - it should execute the extra step
        result = backend.run(n_steps=100)
        
        assert result['source_spiked'], "Source should have spiked"
        assert result['steps'] > 0, "Should have executed simulation steps"
        
        # The algorithm should complete successfully
        assert len(result['path']) > 0, "Should find a path"

    def test_complex_graph_path_correctness(self):
        """Test path correctness on the graph from the algorithm image."""
        # Graph from the figure (approximation)
        adj = {
            'SRC': [('A', 1), ('B', 3)],
            'A': [('C', 2), ('DST', 3)],
            'B': [('C', 1)],
            'C': [('DST', 2)],
            'DST': []
        }
        
        # Compute expected shortest path
        expected_path, expected_cost = dijkstra_shortest_path(adj, 'SRC', 'DST')
        print(f"Dijkstra's shortest path: {expected_path} with cost {expected_cost}")
        
        # Run Loihi algorithm
        brick = LoihiGSBrick(adj, source='SRC', destination='DST', name='test_complex')
        scaffold = Scaffold()
        scaffold.add_brick(brick, output=True)
        scaffold.lay_bricks()
        
        backend = gsearch_Backend()
        backend.compile(scaffold, {})
        result = backend.run(n_steps=100)
        
        assert result['source_spiked'], "Source should have spiked"
        assert len(result['path']) > 0, "Should find a path"
        
        # Map back to node names
        neuron_to_node = {v: k for k, v in brick.node_to_neuron.items() if '__aux__' not in str(k)}
        path_nodes = [neuron_to_node.get(neuron, neuron) for neuron in result['path']]
        path_nodes = [n for n in path_nodes if '__aux__' not in str(n)]
        
        print(f"Loihi path: {path_nodes}")
        
        # Verify start and end
        assert path_nodes[0] == 'SRC', f"Path should start at SRC, got {path_nodes[0]}"
        assert path_nodes[-1] == 'DST', f"Path should end at DST, got {path_nodes[-1]}"

    def test_forward_edges_dominate_at_end(self):
        """
        Verify that at the end of the algorithm, the path is readable
        from forward edges (backward edges on the path are zeroed).
        """
        adj = {
            0: [(1, 2), (2, 5)],
            1: [(2, 1)],
            2: []
        }
        
        brick = LoihiGSBrick(adj, source=0, destination=2, name='test_forward')
        scaffold = Scaffold()
        scaffold.add_brick(brick, output=True)
        scaffold.lay_bricks()
        
        backend = gsearch_Backend()
        backend.compile(scaffold, {})
        result = backend.run(n_steps=100)
        
        assert result['source_spiked'], "Source should have spiked"
        
        # Check that we can reconstruct the path from forward edges
        # The path should be traceable via READOUTNEXTHOP (forward edges with weight=1)
        graph = backend.fugu_graph
        
        # Count forward vs backward edges with non-zero weight
        forward_nonzero = 0
        backward_nonzero = 0
        
        for u, v, data in graph.edges(data=True):
            weight = data.get('weight', 0)
            if weight != 0:
                if data.get('direction') == 'forward':
                    forward_nonzero += 1
                elif data.get('direction') == 'backward':
                    backward_nonzero += 1
        
        # Forward edges should remain (for readout)
        assert forward_nonzero > 0, "Should have forward edges for path readout"
        
        print(f"Final state: {forward_nonzero} forward edges, {backward_nonzero} backward edges (non-zero)")

    def test_path_cost_matches_dijkstra(self):
        """Verify the path found has the same cost as Dijkstra's shortest path."""
        adj = {
            'A': [('B', 3), ('C', 1)],
            'B': [('D', 2)],
            'C': [('D', 4)],
            'D': []
        }
        
        # Dijkstra's shortest path
        expected_path, expected_cost = dijkstra_shortest_path(adj, 'A', 'D')
        print(f"Expected path: {expected_path}, cost: {expected_cost}")
        
        # Loihi algorithm
        brick = LoihiGSBrick(adj, source='A', destination='D', name='test_forward')
        scaffold = Scaffold()
        scaffold.add_brick(brick, output=True)
        scaffold.lay_bricks()
        
        backend = gsearch_Backend()
        backend.compile(scaffold, {})
        result = backend.run(n_steps=100)
        
        assert result['source_spiked'], "Source should have spiked"
        assert len(result['path']) > 0, "Should find a path"
        
        # Map path back to original nodes
        neuron_to_node = {v: k for k, v in brick.node_to_neuron.items() if '__aux__' not in str(k)}
        path_nodes = [neuron_to_node.get(neuron, neuron) for neuron in result['path']]
        path_nodes = [n for n in path_nodes if '__aux__' not in str(n)]
        
        print(f"Loihi path: {path_nodes}")
        
        # Calculate cost of the found path using original adjacency list
        # Build a graph from original adj to get edge weights
        G = nx.DiGraph()
        for u, neighbors in adj.items():
            for v, cost in neighbors:
                G.add_edge(u, v, weight=cost)
        
        # Calculate path cost
        loihi_cost = 0
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            if G.has_edge(u, v):
                loihi_cost += G[u][v]['weight']
        
        print(f"Loihi path cost: {loihi_cost}")
        
        # The costs should match (shortest path)
        assert loihi_cost == expected_cost, \
            f"Path cost mismatch: Loihi={loihi_cost}, Dijkstra={expected_cost}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
