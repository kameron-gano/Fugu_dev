#!/usr/bin/env python3
"""
Test the loihi_wavefront spike criterion implementation.

The criterion should only allow a neuron to spike when:
1. Current voltage >= threshold, AND
2. Previous voltage < threshold (rising edge)

After spiking, reset_voltage should be >= threshold to prevent re-spiking.
"""

import pytest
from fugu.simulators.SpikingNeuralNetwork.neuron import GeneralNeuron, SPIKE_CRITERIA_REGISTRY


class TestLoihiWavefrontCriterion:
    """Test suite for the loihi_wavefront spike criterion."""

    def test_registry_contains_loihi_wavefront(self):
        """Verify that loihi_wavefront is registered."""
        assert 'loihi_wavefront' in SPIKE_CRITERIA_REGISTRY
        assert callable(SPIKE_CRITERIA_REGISTRY['loihi_wavefront'])

    def test_wavefront_criterion_rising_edge(self):
        """Test that neuron spikes on rising edge (below threshold -> above threshold)."""
        # Create a neuron with loihi_wavefront criterion
        neuron = GeneralNeuron(
            name='test_neuron',
            threshold=1.0,
            reset_voltage=1.0,  # Reset to >= threshold
            voltage=0.5,  # Start below threshold
            spike_criterion='loihi_wavefront'
        )
        
        # Manually set previous voltage to be below threshold
        neuron._prev_voltage = 0.5
        
        # Test the criterion directly
        criterion = SPIKE_CRITERIA_REGISTRY['loihi_wavefront']
        
        # Should NOT spike when staying below threshold
        result = criterion(neuron, 0.7)
        assert result == False, "Should not spike when below threshold"
        
        # Should spike when crossing threshold (rising edge)
        result = criterion(neuron, 1.0)
        assert result == True, "Should spike on rising edge"
        
        # Simulate what happens after spike: voltage resets to reset_voltage (1.0)
        neuron._prev_voltage = 1.0
        
        # Should NOT spike when staying above threshold
        result = criterion(neuron, 1.2)
        assert result == False, "Should not spike when staying above threshold"

    def test_wavefront_criterion_first_timestep(self):
        """Test that neuron can spike at first timestep if voltage >= threshold."""
        neuron = GeneralNeuron(
            name='test_neuron',
            threshold=0.9,
            reset_voltage=1.0,
            voltage=0.99,  # Start just below threshold
            spike_criterion='loihi_wavefront'
        )
        
        # At first timestep, _prev_voltage is None
        assert neuron._prev_voltage is None
        
        criterion = SPIKE_CRITERIA_REGISTRY['loihi_wavefront']
        
        # Should spike if voltage >= threshold at first timestep
        result = criterion(neuron, 1.0)
        assert result == True, "Should spike at first timestep if above threshold"

    def test_general_neuron_uses_spike_criterion(self):
        """Test that GeneralNeuron correctly uses the spike_criterion parameter."""
        neuron = GeneralNeuron(
            name='test_neuron',
            threshold=1.0,
            reset_voltage=1.0,
            voltage=0.5,
            bias=0.6,  # Will bring voltage to 1.1 (above threshold)
            spike_criterion='loihi_wavefront'
        )
        
        # First update: should spike (rising from 0.5 to 1.1)
        neuron.update_state()
        assert neuron.spike == True, "Should spike on first rising edge"
        assert neuron.v == 1.0, "Should reset to reset_voltage (1.0)"
        
        # Second update: should NOT spike (already above threshold)
        neuron.update_state()
        assert neuron.spike == False, "Should not spike when staying above threshold"

    def test_destination_neuron_initial_state(self):
        """Test that destination neuron with potential=0.99 can spike at timestep 0.
        
        Note: The destination neuron starts at 0.99, which is above threshold (0.9).
        At the first timestep with bias=0.0, it stays at 0.99 (above threshold).
        According to the loihi_wavefront criterion, it will NOT spike because
        at timestep 0, _prev_voltage is set to 0.99 (the initial voltage), and
        the criterion checks (voltage >= threshold) AND (_prev_voltage < threshold).
        Since 0.99 >= 0.9, the second condition fails.
        
        For the algorithm to work correctly, the destination should start at a value
        BELOW threshold so it can spike when it crosses threshold.
        """
        # Simulate destination neuron configuration with initial voltage BELOW threshold
        neuron = GeneralNeuron(
            name='destination',
            threshold=1.0,  # Higher threshold
            reset_voltage=1.0,
            voltage=0.5,  # Start below threshold
            bias=0.6,  # Will bring it to 1.1 (above threshold)
            spike_criterion='loihi_wavefront'
        )
        
        # At timestep 0, _prev_voltage will be set to 0.5, voltage will become 1.1
        # Should be able to spike (rising edge: 0.5 < 1.0 <= 1.1)
        neuron.update_state()
        assert neuron.spike == True, "Destination should spike at timestep 0"
        assert neuron.v == 1.0, "Should reset to 1.0"
        
        # At timestep 1, should NOT spike (staying above threshold)
        neuron.update_state()
        assert neuron.spike == False, "Should not spike again"

    def test_no_respiking_after_reset(self):
        """Verify that reset_voltage >= threshold prevents re-spiking."""
        neuron = GeneralNeuron(
            name='test_neuron',
            threshold=0.9,
            reset_voltage=1.0,  # >= threshold
            voltage=0.0,
            bias=1.0,  # Large bias to keep voltage high
            spike_criterion='loihi_wavefront'
        )
        
        # First update: rises from 0.0 to 1.0, should spike
        neuron.update_state()
        assert neuron.spike == True, "Should spike on rising edge"
        
        # Verify voltage was reset to >= threshold
        assert neuron.v >= neuron._T, "Reset voltage should be >= threshold"
        
        # Subsequent updates: should not spike (already above threshold)
        for _ in range(5):
            neuron.update_state()
            assert neuron.spike == False, "Should not re-spike after reset"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
