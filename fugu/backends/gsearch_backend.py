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
					print(f"DEBUG zero_backward_edges: Zeroed edge ({post}, {pre})")
		
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
		"""Return next hop by checking which BACKWARD edge was pruned.

		Algorithm from standalone implementation (Lines 14-20):
		For current node i, find node j such that:
		1. There's a forward edge i→j (direction='forward', weight=1)
		2. The backward edge j→i was PRUNED (originally had weight>0, now weight=0)

		This method inspects `self.current_hop` to determine current node.

		Returns:
		    The next neuron name (str) or None if no valid hop.
		"""
		current = getattr(self, 'current_hop', None)
		if current is None or current not in self.fugu_graph:
			return None
		
		# Get bundle to access original backward edge weights
		bundle = self.fugu_graph.graph.get('loihi_gs', {})
		w_backward_original = bundle.get('w_backward_original', {})
		
		# Look for forward edges from current
		for _, nxt, data in self.fugu_graph.out_edges(current, data=True):
			if data.get('direction') == 'forward' and data.get('weight', 0) == 1.0:
				# Check if backward edge nxt→current was pruned
				backward_key = (nxt, current)
				if backward_key in w_backward_original:
					original_weight = w_backward_original[backward_key]
					if original_weight > 0:
						# Check if this backward edge is now zeroed
						if self.fugu_graph.has_edge(nxt, current):
							current_weight = self.fugu_graph[nxt][current].get('weight', 0)
							if current_weight == 0:
								return nxt
		return None

	def reconstruct_path(self) -> list[str]:
		"""
		Reconstruct forward path from source to destination after pruning.

		Assumes all necessary backward edges have been zeroed and remaining
		forward edges (direction=='forward', weight>0) form a tree/unique path.
		Returns list of neuron names from source to destination. Empty list if
		source/destination not known or path not found.
		"""
		bundle = self.fugu_graph.graph.get('loihi_gs') or {}
		src = bundle.get('source_neuron')
		dst = bundle.get('destination_neuron')
		if not src or not dst:
			return []
		if src not in self.fugu_graph or dst not in self.fugu_graph:
			return []
		path = [src]
		self.current_hop = src
		visited = {src}
		# Hard cap to avoid infinite loops on malformed graphs
		limit = len(self.fugu_graph.nodes)
		steps = 0
		while self.current_hop != dst and steps < limit:
			nxt = self.readout_next_hop()
			if nxt is None or nxt in visited:
				return []  # no unique path
			path.append(nxt)
			visited.add(nxt)
			self.current_hop = nxt
			steps += 1
		if self.current_hop != dst:
			return []
		return path


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
				# Record first spike time
				if self.spike_time[name] == -1:
					self.spike_time[name] = self.current_timestep
					newly_spiked.add(name)
					print(f"t={self.current_timestep}: {name} SPIKED (first time)")
		
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
				print(
					f"  EDGE ({post},{pre}) d={delay} post_sim={post_sim} pre_sim={pre_sim} post_alg={post_alg} pre_alg={pre_alg} expect_post_alg={pre_alg - delay} prune={should_prune}"
				)
				if should_prune:
					print(f"    PRUNE: ({post},{pre}) ✓ (post_alg={post_alg} == pre_alg({pre_alg}) - d({delay}))")
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
		print(f"\n=== sim_t={self.current_timestep} (algo_t={self.current_timestep - self.algo_offset}): Initial propagation ===")
		last_diag = self.prune_step()
		source_spiked = last_diag.get('source_spiked', False)

		# Main loop
		while not source_spiked and self.current_timestep < limit:
			self.current_timestep += 1
			self.nn.step()
			algo_t = self.current_timestep - self.algo_offset
			print(f"\n=== sim_t={self.current_timestep} (algo_t={algo_t}) ===")
			last_diag = self.prune_step()
			source_spiked = last_diag.get('source_spiked', False)
		
		# Line 26: One extra wavefront step after source spikes
		if source_spiked and self.current_timestep < limit:
			self.current_timestep += 1
			self.nn.step()
			self.prune_step()

		path = self.reconstruct_path() if source_spiked else []
		return {
			'path': path,
			'steps': self.current_timestep,
			'source_spiked': source_spiked,
			'remaining_backward': len(self.remaining_backward_edges())
		}