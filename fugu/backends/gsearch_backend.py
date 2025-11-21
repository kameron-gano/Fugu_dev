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
		# for all j fanout of i do
		#   if w _{i,j} = 1
		#       return j
		return

	def reconstruct_path(self) -> list[str]:
		"""
		Reconstruct forward path from source to destination after pruning.

		Assumes all necessary backward edges have been zeroed and remaining
		forward edges (direction=='forward', weight>0) form a tree/unique path.
		Returns list of neuron names from source to destination. Empty list if
		source/destination not known or path not found.
		"""
		# path[0] = src
		# while path[n] != destination:
		# path[n+1] = readout_next_hop
		# n = n + 1
		# return path
		return []


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
		# 1. send a spike into destination
		# 2. while the source has not spiked,
		# 3. advance wavefront (self.nn.step)
		# 4. prune_step
		# 4. increment time step
		# 5. advance wavefront one more time (self.nn.step)
		# 6. step one more time
		return self.reconstruct_path()