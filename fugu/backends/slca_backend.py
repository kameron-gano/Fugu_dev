from collections import deque
from warnings import warn

from typing import Optional, Dict, Any
import fugu.simulators.SpikingNeuralNetwork as snn

from .backend import Backend, PortDataIterator
from ..utils.export_utils import results_df_from_dict
from ..utils.misc import CalculateSpikeTimes
from .snn_backend import snn_Backend
import numpy as np

class slca_Backend(snn_Backend):

    def normalize_columns(self, A):
        """ Normalize columns of A to unit norm. """
        norms = np.linalg.norm(A, axis=0)
        norms[norms == 0] = 1.0
        return A / norms

    def compile(self, scaffold, compile_args: Dict[str, Any] = {}, normalize_weights: bool = True):
        """
        Extra compile args (in addition to snn_Backend):
          - Phi : (M x N) dictionary matrix (columns normalized)
          - y   : (M,)     observed patch (flattened)
          - K   : (M x M)  optional blur operator. If provided, Psi = K @ Phi.
                           Otherwise Psi = Phi.
          - lam : float    L1 threshold λ (default 0.1)
          - dt : float     simulation step (default 1e-4)
          - tau_syn : float synaptic time constant (default 1e-2)
          - T_steps : int  total S-LCA steps to run in run(), if not overridden
          - t0_steps : int ignore first t0_steps for tail readout (optional)
          - unit_area : bool  (default True) scale inhibition by 1/tau_syn
        """
        self.Phi = compile_args.get('Phi', None)
        self.y_obs = compile_args.get('y', None)
        self.K = compile_args.get('K', None)
        self.lam = float(compile_args.get('lam', 0.1))
        self.dt = float(compile_args.get('dt', 1e-4))
        self.tau= float(compile_args.get('tau_syn', 1e-2))
        self.T_steps = int(compile_args.get('T_steps', 1000))
        self.t0_steps = int(compile_args.get('t0_steps', max (1, self.T_steps // 10)))
        self.unit_area = bool(compile_args.get('unit_area', True))

        if self.Phi is None or self.y_obs is None:
            raise ValueError("LCA_Backend.compile requires Phi (M x N) and y (M,) in compile_args.")

        if normalize_weights:
            self.Phi = self.normalize_columns(self.Phi)

        if self.K is not None:
            Psi = np.asarray(self.K, dtype=float) @ self.Phi
        else:
            Psi = self.Phi
        
        # LCA constants: b, W (zero diag)
        self.b = Psi.T @ self.y_obs                 # (N,)
        W = Psi.T @ Psi                             # (N x N)
        np.fill_diagonal(W, 0.0)

        # Unit-area exponential synapse scaling (each spike contributes unit area)
        self.W = (W / self.tau) if self.unit_area else W

        # Precompute decay for synaptic traces
        self.decay = float(np.exp(-self.dt / self.tau))

        # Dimensions and external S-LCA states
        self.N = self.Phi.shape[1]
        self.inhibition = np.zeros(self.N)                   # filtered spike traces
        self.soma_current = np.zeros(self.N)                  # soma currents
        self.int_soma_current = np.zeros(self.N)              # ∫ μ dt (for Tλ(u) readout)
        self.spikes_prev = np.zeros(self.N)         # last-step spikes (0/1)

        # Let the parent build the physical SNN (neurons/synapses).
        # We won't rely on presynaptic synapses; we push Δv via bias per step.
        super().compile(scaffold, compile_args)
        for name, n in self.nn.nrns.items():
            if "begin" in name or "complete" in name:
                continue
            n.leakage_constant = 1.0
        # Configure LIF shells: no leak, known threshold/reset

    def slca_step(self):
        # update decays of inhibitory spikes based on spike history
        self.inhibition = self.decay*self.inhibition + self.spikes_prev

        # update soma current
        self.soma_current = self.b - (self.W @ (self.inhibition / self.tau))

        # integrate the change in soma current for this time step
        self.int_soma_current += self.soma_current * self.dt

        # Direct voltage integration: v += dt * (mu - lam)
        for name, n in self.nn.nrns.items():
            if "begin" in name or "complete" in name:
                continue
            # Extract neuron index from name
            idx = int(name.split('_')[-1])
            # Direct S-LCA voltage update
            dv = self.dt * (self.soma_current[idx] - self.lam)
            n.v += dv
            # Check for spike and reset
            if n.v >= n.threshold:
                n.spike_hist.append(True)
                n.v = 0.0  # Reset
            else:
                n.spike_hist.append(False)
        
        # Don't run the neural network - we're handling integration manually

        # Extract spike information from what we just computed
        new_spikes = np.zeros(self.N, dtype=float)
        for name, n in self.nn.nrns.items():  
            if "begin" in name or "complete" in name:
                continue
            idx = int(name.split('_')[-1])
            new_spikes[idx] = 1.0 if (n.spike_hist and n.spike_hist[-1]) else 0.0
        self.spikes_prev = new_spikes

    def run(self, n_steps: Optional[int] = None, return_readout: bool = True):
        """
        Run S-LCA in this backend.

        Args:
          n_steps: number of S-LCA steps (defaults to self.T_steps).
          return_readout:
            - True: return a dict with 'a_tail', 'a_rate', 'counts', 'x_hat'
            - False: mimic snn_Backend.run() and return spike_times dataframe.

        NOTE: We do NOT rely on input spikes; Δv is injected via bias each step.
        """

        steps = int(self.T_steps if n_steps is None else n_steps)

        int_soma_current_at_t0 = None
        self.spikes_prev[:] = 0.0
        self.soma_current[:] = 0.0
        self.int_soma_current[:] = 0.0
        self.spikes_prev[:] = 0.0

        for n in self.nn.nrns.values():
            n.v = 0.0
            n.spike_hist.clear()

        for k in range(steps):
            self.slca_step()
            if int_soma_current_at_t0 is None and k >= self.t0_steps:
                int_soma_current_at_t0 = self.int_soma_current.copy()

        if int_soma_current_at_t0 is None:
            int_soma_current_at_t0 = np.zeros_like(self.int_soma_current)

        T_tail = (steps - self.t0_steps) * self.dt
        mu_tail = (self.int_soma_current - int_soma_current_at_t0) / max(T_tail, 1e-12)
        a_tail = np.maximum(0.0, mu_tail - self.lam)

        counts = np.array([sum(self.nn.nrns[n].spike_hist) for n in self.nn.nrns], dtype=int)
        T_sec = steps * self.dt
        a_rate = counts / max(T_sec, 1e-12)

        x_hat = self.Phi @ a_tail
        return {"a_tail": a_tail, "a_rate": a_rate, "counts": counts, "x_hat": x_hat, "b": self.b, "W": self.W}
