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
		if not self.fugu_graph.has_edge(post, pre):
			return False
		data = self.fugu_graph[post][pre]
		if data.get('direction') != 'backward':
			return False
		if data.get('weight', 0) == 0:
			return False  # already zeroed
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
		count = 0
		for post, pre in edges:
			if self.zero_backward_edge(post, pre):
				count += 1
		return count
	
	def readout_next_hop(self):
		"""Return a deterministic next forward hop from a given current neuron.

		After pruning, each path node should have at most one remaining forward
		edge (direction=='forward', weight>0). To keep backward compatibility with
		the placeholder signature, this method inspects an attribute
		`current_hop` on the backend; callers can set `self.current_hop` before
		invocation. If `current_hop` isn't set, it returns None.

		Returns:
		    The next neuron name (str) or None if no valid hop.
		"""
		current = getattr(self, 'current_hop', None)
		if current is None or current not in self.fugu_graph:
			return None
		candidates = []
		for _, nxt, data in self.fugu_graph.out_edges(current, data=True):
			if data.get('direction') == 'forward' and data.get('weight', 0) > 0:
				candidates.append(nxt)
		if not candidates:
			return None
		# Deterministic selection: pick lowest index (ties arbitrary but stable)
		candidates.sort(key=lambda n: self.fugu_graph.nodes[n].get('index', 1e9))
		return candidates[0]

	def reconstruct_path(self) -> list[str]:
		"""
		Reconstruct forward path from source to destination after pruning.

		After graph reversal, forward edges go from destination toward source.
		We follow forward edges from destination to source, then reverse the path
		to get source -> destination in the original graph orientation.
		
		Returns list of neuron names from source to destination (original graph).
		Empty list if source/destination not known or path not found.
		"""
		bundle = self.fugu_graph.graph.get('loihi_gs') or {}
		src = bundle.get('source_neuron')
		dst = bundle.get('destination_neuron')
		if not src or not dst:
			return []
		if src not in self.fugu_graph or dst not in self.fugu_graph:
			return []
		
		# Start from destination and follow forward edges to source
		path = [dst]
		self.current_hop = dst
		visited = {dst}
		# Hard cap to avoid infinite loops on malformed graphs
		limit = len(self.fugu_graph.nodes)
		steps = 0
		while self.current_hop != src and steps < limit:
			nxt = self.readout_next_hop()
			if nxt is None or nxt in visited:
				return []  # no unique path
			path.append(nxt)
			visited.add(nxt)
			self.current_hop = nxt
			steps += 1
		if self.current_hop != src:
			return []
		
		# Reverse to get source -> destination ordering
		return list(reversed(path))


	def prune_step(self) -> dict[str, Any]:
		"""Perform one graph-search pruning step.

		Detect backward edges whose post neuron spiked this timestep and zero them.
		Returns diagnostics: {'zeroed': int, 'remaining': int, 'source_spiked': bool}.
		Requires that neuron spike history/state be accessible via self.nn.
		"""
		# Build a quick set of spiking neuron names this timestep.
		spiking = {n.name for n in self.nn.nrns.values() if getattr(n, 'spike', False)}
		fired_backward: list[tuple[str, str]] = []
		for post, pre, data in self.fugu_graph.edges(data=True):
			if data.get('direction') == 'backward' and data.get('weight', 0) != 0:
				# backward synapse post->pre fired if post spiked
				if post in spiking:
					fired_backward.append((post, pre))
		zeroed = self.zero_backward_edges(fired_backward)
		remaining = len(self.remaining_backward_edges())
		bundle = self.fugu_graph.graph.get('loihi_gs') or {}
		src = bundle.get('source_neuron')
		source_spiked = src in spiking if src else False
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

		# Prime initial destination spike (Algorithm line 21: s_dst[0] <- 1)
		# Manually trigger destination spike on first step by injecting current
		# We'll add a large bias for one step, then remove it
		dest_neuron = self.nn.nrns[dst]
		original_bias = dest_neuron._b
		dest_neuron._b = 2.0  # Large bias to push above threshold on first step
		
		# Run one step to trigger destination spike
		self.nn.step()
		steps = 1
		
		# Restore original bias
		dest_neuron._b = original_bias
		
		limit = int(n_steps) if n_steps is not None else max(10, 10 * len(self.fugu_graph.nodes))
		source_spiked = self.nn.nrns[src].spike
		last_diag = {'zeroed': 0, 'remaining': len(self.remaining_backward_edges()), 'source_spiked': source_spiked}
		while not source_spiked and steps < limit:
			self.nn.step()
			last_diag = self.prune_step()
			steps += 1
			source_spiked = last_diag.get('source_spiked', False) or self.nn.nrns[src].spike

		path = self.reconstruct_path() if source_spiked else []
		return {
			'path': path,
			'steps': steps,
			'source_spiked': source_spiked,
			'remaining_backward': len(self.remaining_backward_edges())
		}