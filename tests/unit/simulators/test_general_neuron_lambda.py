import numpy as np
import pytest

from fugu.simulators.SpikingNeuralNetwork.neuron import GeneralNeuron, register_spike_criterion

# Ensure deterministic random outcomes for probabilistic spiking
np.random.seed(0)


# Register custom test criteria
@register_spike_criterion('test_always_false')
def always_false_criterion(neuron, voltage):
    """Test criterion that never allows spiking."""
    return False


@register_spike_criterion('test_always_true')
def always_true_criterion(neuron, voltage):
    """Test criterion that always allows spiking."""
    return True


def test_general_neuron_custom_criterion_blocks_spike():
    # Bias large enough to exceed threshold if default logic used
    n = GeneralNeuron(
        name="n",
        threshold=1.0,
        reset_voltage=0.0,
        leakage_constant=1.0,
        voltage=0.0,
        bias=10.0,
        p=1.0,
        record=False,
        compartment={"name": "RecurrentInhibition", "tau_syn": 1.0, "dt": 1.0},
        spike_criterion='test_always_false',
    )

    n.update_state()
    assert n.spike is False, "Custom criterion should block spiking even above threshold"


def test_general_neuron_custom_criterion_allows_spike():
    n = GeneralNeuron(
        name="n2",
        threshold=1000.0,  # very high to ensure default would not spike
        reset_voltage=0.0,
        leakage_constant=1.0,
        voltage=0.0,
        bias=0.5,  # any value
        p=1.0,
        record=False,
        compartment={"name": "RecurrentInhibition", "tau_syn": 1.0, "dt": 1.0},
        spike_criterion='test_always_true',
    )

    n.update_state()
    assert n.spike is True, "Custom criterion should allow spiking regardless of threshold"
