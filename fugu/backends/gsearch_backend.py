from collections import deque
from warnings import warn

from typing import Optional, Dict, Any
import fugu.simulators.SpikingNeuralNetwork as snn

from .backend import Backend, PortDataIterator
from ..utils.export_utils import results_df_from_dict
from ..utils.misc import CalculateSpikeTimes
from .snn_backend import snn_Backend
import numpy as np

class gsearch_Backend(snn_Backend):
	"""
		Backend extension implementing Loihi graph-search runtime operations.
	"""
	def compile(self, scaffold, compile_args):
		super().compile(scaffold, compile_args)
		
		# Store original backward edge weights for readout logic
		bundle = self.fugu_graph.graph.get('loihi_gs', {})
		if 'w_backward_original' not in bundle:
			w_backward_original = {}
			for u, v, data in self.fugu_graph.edges(data=True):
				if data.get('direction') == 'backward':
					# Edge (u, v) is a backward edge, store with key (u, v)
					w_backward_original[(u, v)] = data.get('weight', 0)
			bundle['w_backward_original'] = w_backward_original
			self.fugu_graph.graph['loihi_gs'] = bundle
		
		# Initialize spike_time tracking for all neurons (when they first spiked)
		self.spike_time = {name: -1 for name in self.fugu_graph.nodes()}
		self.current_timestep = 0
		# Parent pointers populated when backward edges are pruned (predecessor 'pre' gets parent 'post')
		# Use plain dict for Python 3.7 compatibility (avoid PEP 585 types)
		self.parent = {}
	
	def zero_backward_edges(self, edges_to_zero: list[tuple[str, str]]) -> int:
		"""Zero out backward edges in both the graph and the neural network.
		
		Args:
		    edges_to_zero: List of (post, pre) tuples representing backward edges post←pre.
		                   The graph stores this as edge (post, pre) with direction='backward'.
		
		Returns:
		    Number of edges successfully zeroed.
		"""
		zeroed_count = 0
		for post, pre in edges_to_zero:
			# The graph stores backward edge post←pre as (post, pre) with direction='backward'
			if self.fugu_graph.has_edge(post, pre):
				edge_data = self.fugu_graph[post][pre]
				if edge_data.get('direction') == 'backward' and edge_data.get('weight', 0) > 0:
					# Zero the weight in the graph
					edge_data['weight'] = 0.0
					
					# Also zero the synapse in the neural network simulator
					# The synapse in nn is from post to pre
					if hasattr(self, 'nn') and hasattr(self.nn, 'syns'):
						syn_key = (post, pre)
						if syn_key in self.nn.syns:
							self.nn.syns[syn_key].wght = 0.0
					
					zeroed_count += 1
		
		return zeroed_count
	
	def remaining_backward_edges(self) -> list[tuple[str, str]]:
		"""Return list of backward edges that still have non-zero weight.
		
		Returns:
		    List of (pre, post) tuples for backward edges with weight > 0.
		"""
		remaining = []
		for u, v, data in self.fugu_graph.edges(data=True):
			if data.get('direction') == 'backward' and data.get('weight', 0) > 0:
				remaining.append((u, v))
		return remaining
	

	
	def readout_next_hop(self):
		"""Return one deterministic next hop among pruned backward edges.

		Selection rule (minimal reconstruction): choose lexicographically first neuron name
		among candidates whose backward edge was pruned. No cost/delay tie-break needed since
		all candidates lie on some shortest path.

		Returns next neuron name or None if no candidate.
		"""
		current = getattr(self, 'current_hop', None)
		if current is None or current not in self.fugu_graph:
			return None
		bundle = self.fugu_graph.graph.get('loihi_gs', {})
		w_backward_original = bundle.get('w_backward_original', {})
		candidates: list[str] = []
		for _, nxt, data in self.fugu_graph.out_edges(current, data=True):
			if data.get('direction') != 'forward' or data.get('weight', 0) != 1.0:
				continue
			bk = (nxt, current)
			if bk not in w_backward_original:
				continue
			if self.fugu_graph.has_edge(nxt, current):
				b_data = self.fugu_graph[nxt][current]
				if b_data.get('direction') == 'backward' and b_data.get('weight', 0) == 0:
					candidates.append(nxt)
		if not candidates:
			return None
		candidates.sort()  # lexicographic deterministic order
		return candidates[0]

	def reconstruct_path(self) -> list[str]:
		"""Reconstruct one shortest path from source to destination.

		1. Try parent-chain reconstruction (fast, deterministic). Parent pointers are assigned
		   the first time a backward edge (post, pre) is pruned; parent[pre] = post.
		2. If parent chain incomplete, fall back to DFS over pruned backward edges choosing
		   lexicographically first candidates with backtracking.
		Returns list of neuron names (source->destination) or empty list if none found.
		"""
		bundle = self.fugu_graph.graph.get('loihi_gs') or {}
		src = bundle.get('source_neuron')
		dst = bundle.get('destination_neuron')
		if not src or not dst:
			return []
		if src not in self.fugu_graph or dst not in self.fugu_graph:
			return []
		# Parent chain attempt
		if src in self.fugu_graph and dst in self.fugu_graph and self.parent:
			chain = [src]
			cur = src
			limit = len(self.fugu_graph.nodes()) + 5
			steps = 0
			while cur != dst and steps < limit:
				if cur not in self.parent:
					break
				cur = self.parent[cur]
				chain.append(cur)
				steps += 1
			if cur == dst:
				return chain
		# Helper to list candidates from a node
		def candidates(node: str) -> list[str]:
			res = []
			w_backward_original = bundle.get('w_backward_original', {})
			for _, nxt, data in self.fugu_graph.out_edges(node, data=True):
				if data.get('direction') != 'forward' or data.get('weight', 0) != 1.0:
					continue
				bk = (nxt, node)
				if bk not in w_backward_original:
					continue
				if self.fugu_graph.has_edge(nxt, node):
					b_data = self.fugu_graph[nxt][node]
					if b_data.get('direction') == 'backward' and b_data.get('weight', 0) == 0:
						res.append(nxt)
			res.sort()
			return res
		# DFS with explicit stack of (node, iterator list, index)
		# Iterative DFS stack entries: (current_node, candidate_list, next_index)
		stack = [(src, candidates(src), 0)]
		path: list[str] = [src]
		visited = {src}
		limit = len(self.fugu_graph.nodes()) + 5
		while stack:
			cur, opts, idx = stack[-1]
			if cur == dst:
				return path
			if idx >= len(opts):
				stack.pop(); path.pop(); continue
			nxt = opts[idx]
			stack[-1] = (cur, opts, idx + 1)
			if nxt in visited:  # avoid cycles
				continue
			next_opts = candidates(nxt)
			stack.append((nxt, next_opts, 0))
			path.append(nxt)
			visited.add(nxt)
			if len(path) > limit:
				break
		return []


	def prune_step(self) -> dict[str, Any]:
		"""Perform one graph-search pruning step (Lines 7-12 of algorithm).

		For each neuron i that JUST spiked at t+1:
		  For each fanin j with backward edge j→i:
		    If j spiked at EXACTLY t - d_{j,i}, prune edge j→i
		
		Returns diagnostics: {'zeroed': int, 'remaining': int, 'source_spiked': bool}.
		"""
		# Get neurons that spiked THIS timestep
		newly_spiked = set()
		for name, neuron in self.nn.nrns.items():
			spike_val = getattr(neuron, 'spike', False)
			if spike_val:
				if self.spike_time[name] == -1:  # first spike time
					self.spike_time[name] = self.current_timestep
					newly_spiked.add(name)
		
		fired_backward: list[tuple[str, str]] = []
		
		# Correct trigger: when a predecessor ("pre") spikes, check its backward edges (post, pre)
		# Because in this encoding post (closer to destination) spikes earlier, predecessor arrives later.
		for pre in newly_spiked:
			# Examine incoming backward edges (post, pre)
			for post, _, data in self.fugu_graph.in_edges(pre, data=True):
				if data.get('direction') != 'backward':
					continue
				# Delay encoded on backward edge (post, pre)
				delay = int(data.get('delay', 1))
				post_sim = self.spike_time.get(post, -1)
				pre_sim  = self.spike_time.get(pre,  -1)
				post_alg = post_sim - self.algo_offset if post_sim >= self.algo_offset else (-1 if post_sim < 0 else 0)
				pre_alg  = pre_sim  - self.algo_offset if pre_sim  >= self.algo_offset else (-1 if pre_sim < 0 else 0)
				# Prune condition: predecessor arrival consistent with earlier post spike
				# post_alg should equal pre_alg - delay
				should_prune = (post_alg >= 0 and pre_alg >= 0 and post_alg == pre_alg - delay)
				if should_prune:
					# Assign parent pointer once: pre's forward successor is post
					if pre not in self.parent:
						self.parent[pre] = post
					fired_backward.append((post, pre))
		
		zeroed = self.zero_backward_edges(fired_backward)
		remaining = len(self.remaining_backward_edges())
		bundle = self.fugu_graph.graph.get('loihi_gs') or {}
		src = bundle.get('source_neuron')
		source_spiked = src in newly_spiked if src else False
		return {'zeroed': zeroed, 'remaining': remaining, 'source_spiked': source_spiked}

	def run(self, n_steps: Optional[int] = None):
		"""Run the Loihi graph-search wavefront until source spikes or timeout.

		Algorithm (simplified):
		 1. Inject an initial spike at the destination neuron.
		 2. Iterate simulation steps:
		    - advance network one step (self.nn.step)
		    - prune backward edges whose post neuron spiked this step
		    - stop if source neuron has spiked
		 3. Reconstruct forward path (source -> destination) via remaining forward edges.

		Args:
		  n_steps: Optional cap on number of simulation iterations. If None, a heuristic
		           limit is chosen (10x number of neurons).

		Returns:
		  dict with keys:
		    'path'              : list[str] path of neuron names source->destination (may be empty)
		    'steps'             : int steps executed
		    'source_spiked'     : bool whether source spiked during run
		    'remaining_backward': int count of backward edges left after pruning
		"""
		bundle = self.fugu_graph.graph.get('loihi_gs') or {}
		dst = bundle.get('destination_neuron')
		src = bundle.get('source_neuron')
		if not src or not dst:
			return {'path': [], 'steps': 0, 'source_spiked': False, 'remaining_backward': len(self.remaining_backward_edges())}
		if dst not in self.nn.nrns or src not in self.nn.nrns:
			return {'path': [], 'steps': 0, 'source_spiked': False, 'remaining_backward': len(self.remaining_backward_edges())}

		# Synthetic initial injection: set destination spike before first step
		dest_neuron = self.nn.nrns[dst]
		dest_neuron.spike = True  # Will be consumed by first nn.step()
		dest_neuron.spike_hist.append(True)

		limit = int(n_steps) if n_steps is not None else max(10, 10 * len(self.fugu_graph.nodes))
		self.current_timestep = 0  # simulator time
		self.algo_offset = 1       # algorithm time = sim_time - 1 (after first step)
		source_spiked = False

		# Perform initial propagation step (algorithm time 0 happens after this)
		self.current_timestep += 1
		self.nn.step()
		# Initial propagation step (destination injection already applied)
		last_diag = self.prune_step()
		source_spiked = last_diag.get('source_spiked', False)

		# Main loop
		while not source_spiked and self.current_timestep < limit:
			self.current_timestep += 1
			self.nn.step()
			last_diag = self.prune_step()
			source_spiked = last_diag.get('source_spiked', False)
		
		if source_spiked and self.current_timestep < limit:
			self.current_timestep += 1
			self.nn.step()
			self.prune_step()

		# Derive shortest-path cost directly from source spike time (algorithm time)
		src_spike_sim = self.spike_time.get(src, -1)
		cost = (src_spike_sim - self.algo_offset) if (source_spiked and src_spike_sim >= self.algo_offset) else float('inf')
		path = self.reconstruct_path() if source_spiked else []
		return {
			'path': path,
			'steps': self.current_timestep,
			'source_spiked': source_spiked,
			'remaining_backward': len(self.remaining_backward_edges()),
			'cost': cost
		}