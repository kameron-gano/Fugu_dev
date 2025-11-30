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
	
	def remaining_backward_edges(self) -> list[tuple[str, str]]:
		"""List (post, pre) neuron-name pairs for backward synapses with weight>0.

		A backward synapse is one whose edge attribute ``direction == 'backward'``.
		Returns:
			list of (post, pre) pairs.
		"""
		out: list[tuple[str, str]] = []
		for post, pre, data in self.fugu_graph.edges(data=True):
			if data.get('direction') == 'backward' and data.get('weight', 0) != 0:
				out.append((post, pre))
		return out

	def zero_backward_edge(self, post: str, pre: str) -> bool:
		"""Set the weight of a backward edge (post->pre) to zero if present.

		Also updates the cached adjacency matrix in ``graph.graph['loihi_gs']['adj_backward']``
		if it exists. Safe no-op if edge missing or not backward.
		Args:
			post: post-synaptic neuron name (destination of backward edge)
			pre:  pre-synaptic neuron name (source of backward edge)
		Returns:
			True if mutated, False otherwise.
		"""
		# Backward edge in graph goes FROM pre TO post
		if not self.fugu_graph.has_edge(pre, post):
			print(f"DEBUG zero_backward_edge: Edge ({pre}, {post}) not in graph")
			return False
		data = self.fugu_graph[pre][post]
		if data.get('direction') != 'backward':
			print(f"DEBUG zero_backward_edge: Edge ({pre}, {post}) is not backward (direction={data.get('direction')})")
			return False
		if data.get('weight', 0) == 0:
			print(f"DEBUG zero_backward_edge: Edge ({pre}, {post}) already zeroed")
			return False  # already zeroed
		print(f"DEBUG zero_backward_edge: Zeroing edge ({pre}, {post}), weight {data.get('weight')} -> 0.0")
		data['weight'] = 0.0
		bundle = self.fugu_graph.graph.get('loihi_gs', {})
		adj = bundle.get('adj_backward')
		# update cached adjacency if present
		if adj is not None:
			i = self.fugu_graph.nodes[pre]['index']
			j = self.fugu_graph.nodes[post]['index']
			adj[j][i] = 0
		return True

	def zero_backward_edges(self, edges: list[tuple[str, str]]) -> int:
		"""Batch zero multiple backward edges.
		Args:
			edges: iterable of (post, pre)
		Returns:
			count of edges successfully zeroed.
		"""
		print(f"DEBUG zero_backward_edges: Called with {len(edges)} edges")
		count = 0
		for post, pre in edges:
			print(f"DEBUG zero_backward_edges: Trying to zero edge post={post}, pre={pre}")
			if self.zero_backward_edge(post, pre):
				count += 1
		print(f"DEBUG zero_backward_edges: Zeroed {count} edges")
		return count
	
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
				print(f"DEBUG prune_step: Neuron {name} has spike={spike_val}, spike_time={self.spike_time.get(name, 'MISSING')}")
				# Record first spike time
				if self.spike_time[name] == -1:
					self.spike_time[name] = self.current_timestep
					newly_spiked.add(name)
					print(f"DEBUG prune_step: Added {name} to newly_spiked (first spike at t={self.current_timestep})")
				else:
					print(f"DEBUG prune_step: Neuron {name} already spiked at t={self.spike_time[name]}, not adding to newly_spiked")
		
		fired_backward: list[tuple[str, str]] = []
		
		# For each neuron i that just spiked
		for post in newly_spiked:
			print(f"DEBUG prune_step: Checking neuron {post}")
			# Check all backward edges coming INTO post (fanin from wavefront direction)
			# Brick creates: graph.add_edge(v, u, direction='backward') for original edge u→v
			# This represents wavefront edge v→u (backward direction)
			# When u spikes, check if v contributed (v→u with delay c)
			for pre, _, data in self.fugu_graph.in_edges(post, data=True):
				print(f"DEBUG prune_step:   In-edge ({pre}, {post}), direction={data.get('direction')}")
				if data.get('direction') == 'backward':
					# Get delay for this edge  
					delay = int(data.get('delay', 1))
					# Fugu neurons spike 1 timestep later than standalone due to LIF dynamics
					# Map Fugu time to standalone time: standalone_t = fugu_t - 1
					# Then check: spike_time[pre] == standalone_t - delay
					# Which is: spike_time[pre] == (fugu_t - 1) - delay
					expected_spike_time = self.current_timestep - delay - 1
					pre_spike_time = self.spike_time.get(pre, -1)
					print(f"DEBUG prune_step:     delay={delay}, expected={expected_spike_time}, actual={pre_spike_time}")
					if pre_spike_time == expected_spike_time:
						print(f"DEBUG prune_step:     MATCH! Appending ({post}, {pre})")
						# Edge in graph is (pre, post), append as (post, pre) for zero_backward_edges
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

		# Prime initial destination spike (time 0) - needed to start wavefront
		dest_neuron = self.nn.nrns[dst]
		dest_neuron.spike = True
		dest_neuron.spike_hist.append(True)

		limit = int(n_steps) if n_steps is not None else max(10, 10 * len(self.fugu_graph.nodes))
		
		# Initialize: destination spikes at t=0 (Line 21)
		# Mark destination as having spiked at t=0
		self.current_timestep = 0
		self.spike_time[dst] = 0
		source_spiked = False
		
		# Lines 22-25: Run until source spikes (starting from t=1)
		while not source_spiked and self.current_timestep < limit:
			self.current_timestep += 1
			self.nn.step()
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