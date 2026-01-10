"""Unit tests for UniversalMemoryTuner."""

import unittest
from caramba.layer.memory_block.memory.tuner import UniversalMemoryTuner
from caramba.layer.memory_block.memory.telemetry import (
    MemoryHealthTelemetry, ResonantSettlingMetrics, VsaNoveltyMetrics
)


class TestUniversalMemoryTuner(unittest.TestCase):
    def test_mode_off_does_nothing(self):
        tuner = UniversalMemoryTuner(mode="off")
        tel = MemoryHealthTelemetry(utilization=0.1)
        report = tuner.update(tel)
        self.assertEqual(report, {})
        self.assertEqual(tuner.resonant_coupling_mult, 1.0)

    def test_memory_starvation_decreases_thresholds_gradually(self):
        # Target utilization is 0.5. Current is 0.1.
        tuner = UniversalMemoryTuner(mode="adaptive", target_utilization=0.5, max_delta_per_step=0.01)
        tel = MemoryHealthTelemetry(utilization=0.1)
        
        # First update: targets change, actuals move by max_delta
        tuner.update(tel)
        self.assertLess(tuner.target_write_threshold, 1.0)
        self.assertAlmostEqual(tuner.write_threshold_mult, 0.99, places=5) # 1.0 - 0.01
        
        # After 10 updates, it should still be moving gradually
        for _ in range(9):
            tuner.update(tel)
            
        self.assertLess(tuner.write_threshold_mult, 0.95)
        self.assertGreater(tuner.write_threshold_mult, 0.8) # Haven't jumped all at once

    def test_resonant_coupling_moves_smoothly(self):
        tuner = UniversalMemoryTuner(mode="adaptive", lever_smoothing=0.9, max_delta_per_step=0.5)
        # Weak coupling: target increases by 5% each update (1.0 -> 1.05)
        res = ResonantSettlingMetrics(final_sim=0.2, convergence_steps=10, energy_drop=0.0, bucket_entropy=1.0, state_drift=0.0)
        tel = MemoryHealthTelemetry(utilization=0.5, resonant=res)
        
        tuner.update(tel)
        self.assertEqual(tuner.target_resonant_coupling, 1.05)
        # Smoothing: 0.9 * 1.0 + 0.1 * 1.05 = 1.005
        self.assertAlmostEqual(tuner.resonant_coupling_mult, 1.005)

    def test_resonant_steps_discrete_gradual(self):
        tuner = UniversalMemoryTuner(mode="adaptive", lever_smoothing=0.5, max_delta_per_step=0.1)
        # Slow convergence -> target_resonant_steps_delta increases by 0.1
        res = ResonantSettlingMetrics(final_sim=0.8, convergence_steps=20, energy_drop=0.0, bucket_entropy=1.0, state_drift=0.0)
        tel = MemoryHealthTelemetry(utilization=0.5, resonant=res)
        
        # It takes multiple updates to increment the integer delta
        for _ in range(5):
            tuner.update(tel)
        self.assertEqual(tuner.resonant_steps_delta, 0) # Still 0 due to rounding and small steps
        
        for _ in range(10):
            tuner.update(tel)
        # eventual increment
        self.assertGreaterEqual(tuner.target_resonant_steps_delta, 1.0)
        self.assertGreaterEqual(tuner.resonant_steps_delta, 0)


if __name__ == "__main__":
    unittest.main()
