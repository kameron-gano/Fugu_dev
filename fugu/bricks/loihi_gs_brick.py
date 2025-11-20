#!/usr/bin/env python3
"""Loihi Graph-Search Brick

This brick implements the preprocessing and mapping described in the
"Advancing Neuromorphic Computing With Loihi" graph-search algorithm.

It accepts a weighted directed graph (adjacency list or adjacency matrix)
and converts it into a Loihi-compatible neuron/synapse representation where
every directed edge (i -> j) is mapped to:
  - a forward synapse i -> j with weight=1 and delay=0 (readout)
  - a backward synapse j -> i with weight=1 and delay=c_{i,j} - 1

Additionally, to satisfy the fan-out constraint, outgoing edges from
branching nodes are given cost 1 and any extra cost is pushed downstream
onto auxiliary single-fanout nodes as described in the paper.

This file provides `LoihiGSBrick` which follows the Fugu `Brick` API.
"""

from typing import Any, Dict, Iterable, List, Tuple
from .bricks import Brick

import networkx as nx


class LoihiGSBrick(Brick):
    """Brick that converts a weighted directed graph into a Loihi graph-search SNN.

    Parameters
    ----------
    input_graph : dict | list | 2D-array | networkx.DiGraph
        The input graph. Supported formats:
          - adjacency list: dict[label] -> iterable of neighbor labels or (neighbor, weight)
          - adjacency matrix: 2D array-like where [i][j] > 0 indicates cost
          - networkx.DiGraph with edge attribute 'weight' or 'cost'
    name : str
        Brick name used to generate neuron names.
    require_integer_costs : bool
        If True, raise on non-integer costs. If False, costs are rounded.
    The input graph is required to be (weakly) connected; this is validated
    on construction/build.
    max_delay : int or None
        Optional clamp for delays.
    """

    def __init__(self,
                 input_graph: Any,
                 name: str = "LoihiGS",
                 require_integer_costs: bool = True,
                 max_delay: int = None):
        super().__init__(name=name)
        self.input_graph = input_graph
        self.require_integer_costs = require_integer_costs
        self.max_delay = max_delay

        # Populated after preprocess
        self.node_list: List[Any] = []
        self.edges: List[Tuple[Any, Any, int]] = []
        self.node_to_neuron: Dict[Any, str] = {}

    # ---- parsing and preprocessing helpers ----
    def _parse_input(self) -> Tuple[List[Any], List[Tuple[Any, Any, int]]]:
        """Parse input_graph into (nodes, edges) where edges are (u, v, cost).

        Supports dict adjacency lists, networkx.DiGraph and dense matrices.
        """
        g = self.input_graph
        nodes = []
        edges: List[Tuple[Any, Any, int]] = []

        # networkx
        if isinstance(g, nx.DiGraph) or isinstance(g, nx.Graph):
            nodes = list(g.nodes())
            for u, v, data in g.edges(data=True):
                c = data.get('weight', data.get('cost', 1))
                edges.append((u, v, int(round(c))))
            return nodes, edges

        # adjacency list (dict)
        if isinstance(g, dict):
            nodes = list(g.keys())
            for u, nbrs in g.items():
                for e in nbrs:
                    if isinstance(e, (list, tuple)) and len(e) >= 2:
                        v, c = e[0], e[1]
                    else:
                        v, c = e, 1
                    if self.require_integer_costs:
                        if not float(c).is_integer():
                            raise ValueError(f"Non-integer cost {c} on edge {u}->{v}")
                        c = int(c)
                    else:
                        c = int(round(float(c)))
                    edges.append((u, v, c))
            # ensure nodes include any referenced nodes
            extra = {v for (_, v, _) in edges} - set(nodes)
            nodes.extend(sorted(list(extra)))
            return nodes, edges

        # adjacency matrix like (list of lists / 2D array)
        try:
            # assume matrix-like
            n = len(g)
            nodes = list(range(n))
            for i in range(n):
                row = g[i]
                for j in range(len(row)):
                    val = row[j]
                    if val:
                        if self.require_integer_costs and not float(val).is_integer():
                            raise ValueError(f"Non-integer cost {val} at [{i}][{j}]")
                        c = int(round(val))
                        edges.append((i, j, c))
            return nodes, edges
        except Exception:
            raise ValueError("Unsupported input_graph format")

    def _preprocess_fanout(self, nodes: List[Any], edges: List[Tuple[Any, Any, int]]):
        """Enforce fan-out constraint by pushing cost onto auxiliary single-fanout nodes.

        For any node i with outdegree > 1, replace outgoing edge (i->j, c>1)
        with (i->u_ij, cost=1) and (u_ij->j, cost=c-1).
        """
        # build adjacency map to compute outdegree
        out_map: Dict[Any, List[Tuple[Any, int]]] = {}
        for u, v, c in edges:
            out_map.setdefault(u, []).append((v, c))

        new_nodes = list(nodes)
        new_edges: List[Tuple[Any, Any, int]] = []

        for u in new_nodes:
            out = out_map.get(u, [])
            if len(out) <= 1:
                # keep edges as-is
                for v, c in out:
                    new_edges.append((u, v, c))
                continue

            # branching node: ensure all outgoing edges leaving u have cost 1
            for v, c in out:
                if c <= 1:
                    new_edges.append((u, v, c))
                else:
                    # create auxiliary node
                    aux = (f"{u}__aux__{v}")
                    # ensure unique
                    k = 1
                    base = aux
                    while aux in new_nodes:
                        aux = f"{base}_{k}"
                        k += 1
                    new_nodes.append(aux)
                    # replace edge
                    new_edges.append((u, aux, 1))
                    new_edges.append((aux, v, c - 1))

        # also add edges from nodes that originally had no outgoing entries
        # (covered by building from original edges above)
        return new_nodes, new_edges

    # ---- mapping to Loihi graph ----
    def _map_to_loihi(self, graph: nx.DiGraph, nodes: List[Any], edges: List[Tuple[Any, Any, int]]):
        """Add neurons and synapses to the provided networkx DiGraph.

        Nodes are added with names generated via `self.generate_neuron_name` and
        edges are added with attributes 'weight' and 'delay' (integers).
        """
        # create neurons
        for idx, n in enumerate(nodes):
            neuron_name = self.generate_neuron_name(str(n))
            self.node_to_neuron[n] = neuron_name
            # default neuron properties (consistent with other bricks)
            graph.add_node(neuron_name,
                           index=idx,
                           threshold=0.9,
                           decay=0,
                           p=1.0,
                           potential=0.0)

        # add synapses (forward + backward)
        for u, v, c in edges:
            if c < 1:
                raise ValueError(f"Edge cost must be >=1, got {c} for {u}->{v}")
            pre = self.node_to_neuron[u]
            post = self.node_to_neuron[v]
            # forward synapse i -> j with delay 0
            graph.add_edge(pre, post, weight=1.0, delay=0)
            # backward synapse j -> i with delay c - 1
            bdelay = c - 1
            if self.max_delay is not None and bdelay > self.max_delay:
                bdelay = int(self.max_delay)
            graph.add_edge(post, pre, weight=1.0, delay=int(bdelay))

    # ---- public API/Brick build ----
    def build(self, graph: nx.DiGraph, metadata, control_nodes, input_lists, input_codings):
        """Build the Loihi graph-search SNN inside the provided networkx DiGraph.

        This method mutates `graph` by adding neurons and synapses. It returns
        the expected Fugu `Brick.build` tuple.
        """
        # parse
        nodes, edges = self._parse_input()

        # require input graph to be (weakly) connected
        tg_in = nx.DiGraph()
        tg_in.add_nodes_from(nodes)
        for u, v, c in edges:
            tg_in.add_edge(u, v)
        if not nx.is_weakly_connected(tg_in):
            raise ValueError("Input graph is not (weakly) connected")

        # preprocess fan-out
        nodes2, edges2 = self._preprocess_fanout(nodes, edges)

        # map to Loihi neuron/synapse network
        self._map_to_loihi(graph, nodes2, edges2)

        # this brick does not provide standard output ports; return minimal structures
        output_lists = [[]]
        output_codings = ['Undefined']
        # control nodes: none by default
        return (graph, {}, [], output_lists, output_codings)


__all__ = ['LoihiGSBrick']
