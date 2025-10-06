from collections import deque
from warnings import warn

import fugu.simulators.SpikingNeuralNetwork as snn

from .backend import Backend, PortDataIterator
from ..utils.export_utils import results_df_from_dict
from ..utils.misc import CalculateSpikeTimes
import snn_Backend
import numpy as np

class slca_Backend(snn_Backend):

    def slca_step():
        # update decays of inhibitory spikes based on spike history
        self.inihibition = self.decay*self.inhibition + self.spikes_prev

        # update soma current
        self.mu = self.b - (self.W @ self.inhibition)

        # integrate the change in soma current for this time step
        self.mu_int += self.mu * self.dt
        
        # hack each LIFNeuron's bias to update its membrane potential
        dV = self.dt * (self.mu - self.lam)
        for i, (name, n) in enumerate(self.nn.nrns.items()):
            n._b = float(dV[i])
        
        # run the nerual network for a single time step
        _ = self.nn.run(n_steps=1, debug_mode=self.debug_mode, record_potentials=False)

        # now we need to figure out what neurons spiked
        new_spikes = np.zeros(self.N, dtype=float)
        for i, (name, n) in enumerate(self.nn.nrns.items()):
            new_spikes = 1.0 if (len(n.spike_hist) and n.spike_hist[-1]) else 0.0
        self.spikes_prev = new_spikes

