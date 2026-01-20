"""Unit tests for UniversalMemoryTuner."""

import unittest
from layer.memory_block.memory.tuner import UniversalMemoryTuner, ParameterExplorer
from layer.memory_block.memory.telemetry import (
    MemoryHealthTelemetry, ResonantSettlingMetrics, VsaNoveltyMetrics
)


class TestParameterExplorer(unittest.TestCase):
    def test_improvement_accelerates_momentum(self):
        explorer = ParameterExplorer(1.0, 0.0, 10.0, step_size=0.05, max_momentum=5)
        initial_momentum = int(explorer.momentum)
        explorer.step(improved=True)
        self.assertGreater(int(explorer.momentum), initial_momentum)

    def test_degradation_can_deactivate(self):
        explorer = ParameterExplorer(1.0, 0.0, 10.0, step_size=0.05, patience=1, cooldown=10, max_momentum=2)
        # Force some momentum first
        explorer.step(improved=True)
        self.assertTrue(explorer.active)
        # Now degrade until it deactivates (patience=1 makes this fast)
        for _ in range(10):
            explorer.step(improved=False)
            if not explorer.active:
                break
        self.assertFalse(explorer.active)

    def test_bounds_are_respected(self):
        explorer = ParameterExplorer(9.9, 0.0, 10.0, step_size=0.5, max_momentum=20)
        explorer.direction = 1.0

        # Should clamp to max
        explorer.step(improved=True)
        self.assertEqual(explorer.value, 10.0)


class TestUniversalMemoryTuner(unittest.TestCase):
    def test_mode_off_does_nothing(self):
        tuner = UniversalMemoryTuner(mode="off")
        tel = MemoryHealthTelemetry(step=0, utilization=0.1)
        report = tuner.update(tel)
        self.assertEqual(report, {})
        # When off, the tuner should not advance step count or emit reports.
        self.assertEqual(int(tuner.step_count), 0)

    def test_warmup_period_no_exploration(self):
        tuner = UniversalMemoryTuner(mode="adaptive", warmup_steps=10)
        tel = MemoryHealthTelemetry(step=0, utilization=0.5)

        # During warmup, parameters shouldn't change
        initial_coupling = tuner.resonant_coupling_mult
        for i in range(10):
            tel.step = i
            tuner.update(tel)

        self.assertEqual(tuner.resonant_coupling_mult, initial_coupling)

    def test_exploration_after_warmup(self):
        tuner = UniversalMemoryTuner(mode="adaptive", warmup_steps=5)

        # Create telemetry that will produce improving objective
        res = ResonantSettlingMetrics(
            final_sim=0.8,
            convergence_steps=5,  # Low steps = good
            energy_drop=0.0,
            bucket_entropy=1.0,
            state_drift=0.0
        )
        vsa = VsaNoveltyMetrics(
            novelty_ema=0.5,
            write_rejection_rate=0.4,  # Target range
            match_confidence=0.8,
            tag_collision_rate=0.1
        )

        # Warmup
        for i in range(6):
            tel = MemoryHealthTelemetry(
                step=i,
                utilization=0.6,  # Good utilization
                conflict_rate=0.1,  # Low conflict
                resonant=res,
                vsa=vsa
            )
            tuner.update(tel)

        # After warmup, parameters should start exploring
        initial_coupling = tuner.resonant_coupling_mult

        # Run more steps
        for i in range(6, 20):
            tel = MemoryHealthTelemetry(
                step=i,
                utilization=0.6,
                conflict_rate=0.1,
                resonant=res,
                vsa=vsa
            )
            tuner.update(tel)

        # At least one parameter should have changed
        changed_conditions = [
            tuner.resonant_coupling_mult != initial_coupling,
            tuner.resonant_damping_mult != tuner.explorers["damping"].value,
            tuner.vsa_novelty_mult != tuner.explorers["novelty"].value,
        ]
        self.assertTrue(any(changed_conditions), "Parameters should explore after warmup")

    def test_objective_function_computed(self):
        tuner = UniversalMemoryTuner(mode="adaptive")

        res = ResonantSettlingMetrics(
            final_sim=0.8,
            convergence_steps=5,
            energy_drop=0.0,
            bucket_entropy=1.0,
            state_drift=0.0
        )
        vsa = VsaNoveltyMetrics(
            novelty_ema=0.5,
            write_rejection_rate=0.4,
            match_confidence=0.8,
            tag_collision_rate=0.1
        )

        tel = MemoryHealthTelemetry(
            step=0,
            utilization=0.6,
            conflict_rate=0.1,
            resonant=res,
            vsa=vsa
        )

        report = tuner.update(tel)

        # Should have objective in report
        self.assertIn("tuner/objective", report)
        self.assertIsInstance(report["tuner/objective"], float)

    def test_visualization_data(self):
        tuner = UniversalMemoryTuner(mode="adaptive")

        viz = tuner.get_viz_data()

        # Should have all parameters
        self.assertIn("Coupling", viz)
        self.assertIn("Damping", viz)
        self.assertIn("Novelty", viz)
        self.assertIn("Threshold", viz)
        self.assertIn("Steps Δ", viz)

        # Each should have actual, target, velocity
        for param_data in viz.values():
            self.assertIn("actual", param_data)
            self.assertIn("target", param_data)
            self.assertIn("velocity", param_data)


if __name__ == "__main__":
    unittest.main()
