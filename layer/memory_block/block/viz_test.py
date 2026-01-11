"""Tests for auto-tuning visualization configuration and triggers."""

import unittest
from unittest.mock import patch, MagicMock
import torch
from caramba.config.layer import MemoryBlockLayerConfig
from caramba.layer.memory_block.block.layer import MemoryBlockLayer

from dataclasses import dataclass, field

@dataclass
class MockCtx:
    step: int = 0
    memblock_collect_aux: bool = False
    memblock_aux_out: dict = field(default_factory=dict)

class TestVisualizationConfig(unittest.TestCase):
    def setUp(self):
        self.cfg = MemoryBlockLayerConfig(
            d_model=128,
            mem_hashes=1,
            mem_autotune="adaptive",
            mem_autotune_viz=True,
            mem_autotune_viz_interval=10
        )
        self.layer = MemoryBlockLayer(self.cfg)

    def test_viz_disabled_impact(self):
        """Verify that when viz is disabled, collect_aux is not forced."""
        self.cfg.mem_autotune_viz = False
        x = torch.randn(1, 1, 128)
        ctx = MockCtx(step=0)
        
        dummy_routing = {
            "idx_r": torch.zeros(1, 1, 1, dtype=torch.long),
            "idx_w": torch.zeros(1, 1, 1, dtype=torch.long),
            "collect_aux": False
        }
        with patch.object(self.layer.memory, 'compute_routing', return_value=dummy_routing) as mock_routing:
            self.layer(x, ctx=ctx)
            _, kwargs = mock_routing.call_args
            self.assertFalse(kwargs['collect_aux'])

    def test_viz_enabled_warmup(self):
        """Verify that during warmup (steps 0-5), collect_aux is forced to True."""
        self.cfg.mem_autotune_viz = True
        x = torch.randn(1, 1, 128)
        
        dummy_routing = {
            "idx_r": torch.zeros(1, 1, 1, dtype=torch.long),
            "idx_w": torch.zeros(1, 1, 1, dtype=torch.long),
            "collect_aux": True
        }
        for step in range(6):
            ctx = MockCtx(step=step)
            with patch.object(self.layer.memory, 'compute_routing', return_value=dummy_routing) as mock_routing:
                self.layer(x, ctx=ctx)
                _, kwargs = mock_routing.call_args
                self.assertTrue(kwargs['collect_aux'])

    def test_viz_enabled_interval(self):
        """Verify that on viz intervals, collect_aux is forced to True."""
        self.cfg.mem_autotune_viz = True
        self.cfg.mem_autotune_viz_interval = 10
        x = torch.randn(1, 1, 128)
        
        dummy_routing = {
            "idx_r": torch.zeros(1, 1, 1, dtype=torch.long),
            "idx_w": torch.zeros(1, 1, 1, dtype=torch.long),
            "collect_aux": True
        }
        # Step 10: should be True
        ctx = MockCtx(step=10)
        with patch.object(self.layer.memory, 'compute_routing', return_value=dummy_routing) as mock_routing:
            self.layer(x, ctx=ctx)
            _, kwargs = mock_routing.call_args
            self.assertTrue(kwargs['collect_aux'])

        # Step 11: should be False
        ctx = MockCtx(step=11)
        # Use False for dummy_routing to match expectation
        dummy_routing_false = dummy_routing.copy()
        dummy_routing_false["collect_aux"] = False
        with patch.object(self.layer.memory, 'compute_routing', return_value=dummy_routing_false) as mock_routing:
            self.layer(x, ctx=ctx)
            _, kwargs = mock_routing.call_args
            self.assertFalse(kwargs['collect_aux'])

    @patch("caramba.layer.memory_block.memory.memory.logger.tuner_status")
    def test_viz_rendering_trigger(self, mock_tuner_status):
        """Verify that logger.tuner_status is called only when enabled and on interval."""
        # 1. Enabled + Interval 1
        self.cfg.mem_autotune_viz = True
        self.cfg.mem_autotune_viz_interval = 1
        
        ctx = MockCtx(step=1)
        x = torch.randn(1, 1, 128)
        
        from caramba.layer.memory_block.memory.tuner import UniversalMemoryTuner
        tuner = UniversalMemoryTuner(mode="adaptive")
        
        dummy_routing = {
            "idx_r": torch.zeros(1, 1, 1, dtype=torch.long),
            "idx_w": torch.zeros(1, 1, 1, dtype=torch.long),
            "collect_aux": True
        }
        
        with patch("caramba.layer.memory_block.memory.tuner.get_shared_tuner", return_value=tuner):
            with patch.object(self.layer.memory, 'collect_health_telemetry') as mock_tel:
                from caramba.layer.memory_block.memory.telemetry import MemoryHealthTelemetry
                mock_tel.return_value = MemoryHealthTelemetry(utilization=0.5)
                
                with patch.object(self.layer.memory, 'compute_routing', return_value=dummy_routing):
                    self.layer(x, ctx=ctx)
                    self.assertTrue(mock_tuner_status.called)

if __name__ == "__main__":
    unittest.main()
