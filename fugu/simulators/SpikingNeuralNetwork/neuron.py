#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numbers
import sys

import numpy as np

from fugu.utils.types import bool_types, float_types, str_types
from fugu.utils.validation import int_to_float, validate_type

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta("ABC", (), {"__slots__": ()})


class Neuron(ABC):
    """
    Abstract Base Class for Neurons. This class defines the minimum set of
    properties of a Neuron.
    """

    @abc.abstractmethod
    def __init__(self, name=None, spike=False):
        """
        Constructor for a Base Neuron class

        Parameters:
            name (any): String, optional.  Neuron name as a string. The default is None.
            spike (bool): Bool, optional.  Spike state of the Neuron. The default is False.

        Returns:
            None
        """

        self.name = name
        self.spike = False
        self.spike_hist = []

    @abc.abstractmethod
    def update_state(self):
        """
        Update the time evolution of the neuron state
        """


class LIFNeuron(Neuron):
    """
    Leaky Integrate and Fire neuron class. LIFNeurons inherit from base Neuron
    class. LIFNeurons inetgrate the incoming signals (weighted sum of spikes
    from pre-synapses). The leak-rate detemines the time evolution of the
    menbrane Voltage. If the voltage exceeds a threshold, the neurons spikes
    (with probability p) and resets to its reset-voltage.
    """

    def __init__(
        self,
        name=None,
        threshold=0.0,
        reset_voltage=0.0,
        leakage_constant=1.0,
        voltage=0.0,
        bias=0.0,
        p=1.0,
        record=False,
    ):
        """
        Constructor for LIFNeurons. Inherits from Neuron Base Class

        Parameters:
            name (any): String, optional.  String name of a neuron. The default is None.
            threshold : Double, optional.  Threshold value above while the neuron spikes. The default is 0.0.
            reset_voltage : Double, optional.  The voltage to which the neuron resets after spiking. The default is 0.0.
            leakage_constant : Double, optional
                The rate at which the neuron voltage decays. The leakage with rate
                m is calculated as m*v. A rate of m=1 indicates no leak. For
                realistic models, 0<= m <=1. The default is 1.0.
            voltage : Double, optional.  Internal voltage of the neuron. The default is 0.0.
            bias : Double, optional. Constant bias voltage value that is added at every timestep. The default is 0.0
            p (double): optional.  Probability of spiking if voltage exceeds threshold.
                p=1 indicates a deterministic neuron. The default is 1.0.
            record (bool): optional.  Indicates if a neuron spike state should be sensed with probes. Default is False.

        Returns:
            none
        """

        threshold = int_to_float(threshold)
        reset_voltage = int_to_float(reset_voltage)
        leakage_constant = int_to_float(leakage_constant)
        voltage = int_to_float(voltage)
        bias = int_to_float(bias)
        p = int_to_float(p)

        validate_type(name, str_types)
        validate_type(threshold, float_types)
        validate_type(reset_voltage, float_types)
        validate_type(leakage_constant, float_types)
        validate_type(voltage, float_types)
        validate_type(bias, float_types)
        validate_type(p, float_types)
        validate_type(record, bool_types)

        if leakage_constant < 0 or leakage_constant > 1:
            raise UserWarning("For realistic models, leakage m should be in the interval [0, 1].")

        if p < 0 or p > 1:
            raise ValueError("Probability p must be in the interval [0, 1].")

        super(LIFNeuron, self).__init__()
        self.name = name
        self._T = threshold
        self._R = reset_voltage
        self._m = leakage_constant
        self._b = bias
        self.v = voltage
        self.presyn = set()
        self.record = record
        self.prob = p

    def update_state(self):
        """
        Updates the time evolution of the states for one time step.
        The input signals are integrated and accumulates with the internal voltage.
        If the internal voltage exceeds the threshold, the neuron spikes and resets.
        Otherwise, the neruon leaks at a fixed rate down to its reset value.
        The neuron spikes with probability p if it exceeds the threshold.

        Returns:
            None
        """
        """Update the states for one time step"""

        input_v = 0.0
        if self.presyn:
            for s in self.presyn:
                if len(s._hist) > 0:
                    input_v += s._hist[0]

        self.v = self.v + input_v + self._b

        if self.v > self._T:
            if np.random.random(1) <= self.prob:
                self.spike = True
                self.v = self._R
            else:
                self.spike = False
                self.v = self._m * self.v
        else:
            self.spike = False
            self.v = self._m * self.v

        self.spike_hist.append(self.spike)

    def show_state(self):
        """
        Display the voltage and spike state of a neuron.
        Return:
            none
        """

        print("Neuron {0}: {1} volts, spike = {2}".format(self.name, self.v, self.spike))

    def show_params(self):
        """
        Display the parameters of a neuron (name, threshold, reset, leak)

        Returns:
            none

        """

        print(
            "Neuron '{0}':\n"
            "Threshold\t  :{1:2} volts,\n"
            "Reset voltage\t  :{2:1} volts,\n"
            "Leakage Constant :{3}\n"
            "Bias :{4}\n".format(self.name, self._T, self._R, self._m, self._b)
        )

    def show_presynapses(self):
        """
        Display the synapses the feed into the neuron.

        Returns:
            none
        """

        if len(self.presyn) == 0:
            print("Neuron {0} receives no external input".format(self.name))
        elif len(self.presyn) == 1:
            print("{0} receives input via synapse: {1}".format(self.__repr__(), self.presyn))
        else:
            print("{0} receives input via synapses: {1}".format(self.__repr__(), self.presyn))

    @property
    def threshold(self):
        return self._T

    @threshold.setter
    def threshold(self, new_threshold):
        new_threshold = int_to_float(new_threshold)
        validate_type(new_threshold, float_types)
        self._T = new_threshold

    @property
    def reset_voltage(self):
        return self._R

    @reset_voltage.setter
    def reset_voltage(self, new_reset_v):
        new_reset_v = int_to_float(new_reset_v)
        validate_type(new_reset_v, float_types)
        self._R = new_reset_v

    @property
    def leakage_constant(self):
        return self._m

    @leakage_constant.setter
    def leakage_constant(self, new_leak_const):
        new_leak_const = int_to_float(new_leak_const)
        validate_type(new_leak_const, float_types)
        self._m = new_leak_const

    @property
    def voltage(self):
        return self.v

    def __str__(self):
        return "LIFNeuron {0}({1}, {2}, {3})".format(self.name, self._T, self._R, self._m)

    def __repr__(self):
        return "LIFNeuron {0}".format(self.name)


class InputNeuron(Neuron):
    """
    Input Neuron. Inherits from class Neuron.
    Input Neurons can read inputs and convert them to spike streams
    """

    def __init__(self, name=None, threshold=0.1, voltage=0.0, record=False):
        """
        Constructor for Input Neuron.

        Parameters:
            name : String, optional. Input neuron name. The default is None.
            threshold (double): optional. Threashold value above which the neuron spikes. The default is 0.1.
            voltage (double): optional. Membrane voltage. The default is 0.0.
            record (bool): optional. Indicates if a neuron spike state should be sensed with probes. The default is False.

        Returns:
            none
        """

        threshold = int_to_float(threshold)
        voltage = int_to_float(voltage)

        validate_type(name, str_types)
        validate_type(threshold, float_types)
        validate_type(voltage, float_types)
        validate_type(record, bool_types)

        super(InputNeuron, self).__init__()
        self.name = name
        self._T = threshold
        self.v = voltage
        self._it = None
        self.record = record

    def connect_to_input(self, in_stream):
        """
        Enables a neuron to read in an input stream of data.

        Parameters:
            in_stream (any): interable data streams of ints or floats. input data (any interable stream such as lists, arrays, etc.).
        Raises:
            TypeError: if in_stream is not iterable.
        Returns:
            None
        """

        if not hasattr(in_stream, "__iter__"):
            raise TypeError("{in_stream} must be iterable".format(**locals()))
        else:
            self._it = iter(in_stream)

    def update_state(self):
        """
        Updates the neuron states. The neuron spikes if the input value in
        the current iteration is above the threshold and resets.

        Raises:
            TypeError: if the input data is not int or float.
        Returns:
            None
        """

        try:
            n = next(self._it)
            if not isinstance(n, numbers.Real):
                raise TypeError("Inputs must be int or float")

            self.v = n

            if self.v > self._T:
                self.spike = True
                self.v = 0
            else:
                self.spike = False
                self.v = 0
        except StopIteration:
            self.spike = False
            self.v = 0

    @property
    def threshold(self):
        return self._T

    @threshold.setter
    def threshold(self, new_threshold):
        new_threshold = int_to_float(new_threshold)
        validate_type(new_threshold, float_types)
        self._T = new_threshold

    @property
    def voltage(self):
        return self.v

    def __str__(self):
        return "InputNeuron {self.name}".format(**locals())

    def __repr__(self):
        return "InputNeuron {self.name}".format(**locals())


if __name__ == "__main__":
    print("Testing LIF Neuron:")
    print("Trying to set probability > 1")
    try:
        n1 = LIFNeuron("n1", 0.5, 0, 0.6, 0, p=1.2)
    except:
        print("Raises type error since probability was greater than 1")

    n1 = LIFNeuron("n1", threshold=1.2, reset_voltage=0.0, leakage_constant=0.6, voltage=1, p=1)
    print("Neuron with intial v = 1; leakage_constant=0.6:")
    print("Timestep 0:")
    n1.show_state()
    print("Timestep 1:")
    n1.update_state()
    n1.show_state()
    print()

    # Input Neuron
    print("Testing Input Neuron")
    n0 = InputNeuron("n0", threshold=0.1)
    try:
        n0.connect_to_input(2)
    except:
        print("Raises TypeError because input is not iterable")
    ip = [5, 4, 0, 1]
    n0.connect_to_input(ip)
    print(f"Input Neuron Spikes for 7 time steps with input stream {ip}:")
    for i, _ in enumerate(range(7)):
        n0.update_state()
        print(f"Time {i}: {n0.spike}")

class CompetitiveNeuron(LIFNeuron):
    """
    Competitive neuron implementing Sparse Local Competitive Algorithm (S-LCA) dynamics.
    
    Implements the following methodology:
    
    Let the soma current at time t be:
    
        u_i[t] = b_i - sum_{j != i} w_ij * r_j[t]
    
    where:
        - b_i is the bias (feedforward drive) for neuron i
        - w_ij is the lateral inhibitory weight from neuron j to neuron i  
        - r_j[t] is the synaptic trace of neuron j at time t
    
    The synaptic trace follows exponential dynamics:
    
        r_j[t] = decay * r_j[t-1] + spike_j[t-1]
    
    where:
        - decay = exp(-dt/tau_syn) is the synaptic decay factor
        - spike_j[t-1] is the spike of neuron j at previous time step
        - tau_syn is the synaptic time constant
    
    The membrane voltage evolves as:
    
        dv_i/dt = u_i[t] - lambda
    
    where lambda is the sparsity threshold. The neuron spikes when v_i >= threshold
    and resets to 0.
    """

    def __init__(
        self,
        name=None,
        threshold=1.0,
        reset_voltage=0.0,
        voltage=0.0,
        bias=0.0,
        dt=1e-3,
        tau_syn=1.0,
        lam=0.1,
        record=False,
    ):
        """
        Constructor for CompetitiveNeuron (S-LCA dynamics).

        Parameters:
            name (str): Neuron name. Default is None.
            threshold (float): Spike threshold. Default is 1.0.
            reset_voltage (float): Voltage after spike (usually 0.0). Default is 0.0.
            voltage (float): Initial membrane voltage. Default is 0.0.
            bias (float): Feedforward bias current b_i. Default is 0.0.
            dt (float): Time step for integration. Default is 1e-3.
            tau_syn (float): Synaptic time constant. Default is 1.0.
            lam (float): Sparsity threshold lambda. Default is 0.1.
            record (bool): Whether to record for probes. Default is False.
        """
        
        # Convert inputs to float
        threshold = int_to_float(threshold)
        reset_voltage = int_to_float(reset_voltage)
        voltage = int_to_float(voltage)
        bias = int_to_float(bias)
        dt = int_to_float(dt)
        tau_syn = int_to_float(tau_syn)
        lam = int_to_float(lam)
        
        # Validate types
        validate_type(name, str_types)
        validate_type(threshold, float_types)
        validate_type(reset_voltage, float_types)
        validate_type(voltage, float_types)
        validate_type(bias, float_types)
        validate_type(dt, float_types)
        validate_type(tau_syn, float_types)
        validate_type(lam, float_types)
        validate_type(record, bool_types)
        
        # Initialize parent class with no leakage (leakage_constant=1.0)
        # and deterministic spiking (p=1.0)
        super(CompetitiveNeuron, self).__init__(
            name=name,
            threshold=threshold,
            reset_voltage=reset_voltage,
            leakage_constant=1.0,  # No leak - pure integrator
            voltage=voltage,
            bias=bias,
            p=1.0,  # Deterministic spiking
            record=record
        )
        
        # S-LCA specific parameters
        self.dt = dt
        self.tau_syn = tau_syn
        self.lam = lam
        self.decay = np.exp(-dt / tau_syn)  # Synaptic decay factor
        
        # State variables for S-LCA
        self.soma_current = 0.0  # u_i[t]
        self.lateral_inhibition = 0.0  # Sum of weighted synaptic traces

    def update_state(self):
        """
        Updates S-LCA neuron state for one time step.
        
        Implements:
        1. Update synaptic traces: r_j[t] = decay * r_j[t-1] + spike_j[t-1]
        2. Compute soma current: u_i[t] = b_i - sum_{j != i} w_ij * r_j[t]
        3. Integrate voltage: dv_i/dt = u_i[t] - lambda
        4. Check for spike and reset

        Returns:
            None
        """
        
        # collect synaptic inputs (weighted spikes from other neurons)
        synaptic_input = 0.0
        if self.presyn:
            for s in self.presyn:
                if len(s._hist) > 0:
                    synaptic_input += s._hist[0]  # This includes weights from synapses

        # r_j[t] = decay * r_j[t-1] + spike_j[t-1]
        self.lateral_inhibition = self.decay * self.lateral_inhibition + synaptic_input


        # u_i[t] = b_i - lateral_inhibition (synaptic inputs are already weighted)
        self.soma_current = self._b - self.lateral_inhibition

        # dv_i/dt = u_i[t] - lambda
        dv = self.dt * (self.soma_current - self.lam)
        self.v += dv
        
        # spike detection and reset
        if self.v >= self._T:
            self.spike = True
            self.v = self._R  # Reset voltage (typically 0.0)
        else:
            self.spike = False
            # No leakage for S-LCA - pure integrator

        # Step 6: Record spike history
        self.spike_hist.append(self.spike)

"""
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
"""