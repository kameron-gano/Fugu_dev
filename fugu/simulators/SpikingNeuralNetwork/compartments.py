#!/usr/bin/env python3
import abc
from typing import Callable

import numpy as np

from fugu.utils.validation import int_to_float, validate_type
from fugu.utils.types import float_types


class Dendrite(abc.ABC):
    """Abstract base class for dendritic compartments.

    A dendritic compartment transforms presynaptic (already weighted) spike
    inputs into an inhibition (or modulation) current made available to the
    soma each timestep. Concrete subclasses must implement ``step`` and
    ``update`` to expose their internal state.
    """

    @abc.abstractmethod
    def step(self, presyn_current: float) -> float:  # pragma: no cover - interface
        """Advance internal state given summed presynaptic current.

        Args:
            presyn_current (float): Summed weighted presynaptic spikes.
        Returns:
            float: Updated inhibition (or modulation) current.
        """
        raise NotImplementedError

    def reset(self):  # optional hook
        pass

    @abc.abstractmethod
    def update(self, neuron, lif_update):  # pragma: no cover - interface
        """Integrate compartment effect with neuron.

        Subclasses decide how to:
          * read presynaptic spikes
          * advance internal state
          * apply modulation to the neuron's LIF step (via lif_update callback)

        Args:
            neuron (GeneralNeuron): The neuron being updated.
            lif_update (Callable): Helper to perform one LIF update with optional overrides.
        """
        raise NotImplementedError


class RecurrentInhibition(Dendrite):
    """Concrete compartment implementing exponential decay recurrent inhibition.

    Dynamics:
        trace[t] = exp(-dt/tau_syn) * trace[t-1] + presyn_current[t-1]
    inhibition(t) == trace[t]
    """

    def __init__(self, tau_syn: float = 1.0, dt: float = 1.0):
        tau_syn = int_to_float(tau_syn)
        dt = int_to_float(dt)
        validate_type(tau_syn, float_types)
        validate_type(dt, float_types)
        if tau_syn <= 0:
            raise ValueError("tau_syn must be positive")
        if dt < 0:
            raise ValueError("dt must be non-negative")
        self.tau_syn = tau_syn
        self.dt = dt
        self.decay = np.exp(-dt / tau_syn)
        self._trace = 0.0

    def step(self, presyn_current: float) -> float:
        presyn_current = int_to_float(presyn_current)
        validate_type(presyn_current, float_types)
        self._trace = self.decay * self._trace + presyn_current
        return self._trace

    @property
    def inhibition(self) -> float:
        return self._trace

    def update(self, neuron, lif_update):
        """Apply recurrent inhibition as subtractive bias before LIF update."""
        synaptic_input = 0.0
        if neuron.presyn:
            for s in neuron.presyn:
                if len(s._hist) > 0:
                    synaptic_input += s._hist[0]

        # Advance trace
        self.step(synaptic_input)
        # Expose diagnostics for LCA
        try:
            neuron.lateral_inhibition = self.inhibition
        except Exception:
            pass
        b_eff = neuron._b - self.inhibition
        try:
            neuron.soma_current = b_eff
        except Exception:
            pass
        lif_update(bias=b_eff, ignore_presyn=True)

    def reset(self):
        self._trace = 0.0

    def __repr__(self):
        return f"RecurrentInhibition(tau_syn={self.tau_syn}, dt={self.dt})"


# Registry of available dendritic compartments by name
COMPARTMENT_REGISTRY = {
    "RecurrentInhibition": RecurrentInhibition,
}

__all__ = ["Dendrite", "RecurrentInhibition", "COMPARTMENT_REGISTRY"]
