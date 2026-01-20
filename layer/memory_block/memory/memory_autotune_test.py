"""Integration test for MemoryBlockMemory auto-tuning."""

import unittest
import torch
from config.layer import MemoryBlockLayerConfig, LayerType
from layer.memory_block.memory.memory import MemoryBlockMemory
from layer.memory_block.state import MemoryBlockState
from layer.memory_block.memory.tuner import UniversalMemoryTuner


class TestMemoryAutotuneIntegration(unittest.TestCase):
    def test_memory_forward_emits_tuning_telemetry(self) -> None:
        torch.manual_seed(42)
        B, T, D = 2, 4, 32

        config = MemoryBlockLayerConfig(
            type=LayerType.MEMORY_BLOCK,
            d_model=D,
            mem_router="resonant",
            mem_autotune="adaptive",
            mem_buckets=8,
            mem_hashes=1,
            mem_assoc=2,
            mem_key_dim=16,
        )

        mem = MemoryBlockMemory(config, D)
        tuner = mem.tuner
        self.assertIsNotNone(tuner)
        assert isinstance(tuner, UniversalMemoryTuner)
        self.assertEqual(tuner.mode, "adaptive")

        u = torch.randn((B, T, D))
        st = MemoryBlockState(
            conv_buf=torch.zeros((B, 0, D)),
            s=torch.zeros((B, 1, D)),
            regs=None,
            step=0,
            mem_k=torch.zeros((B, 1, 8, 2, 16)),
            mem_v=torch.zeros((B, 1, 8, 2, 256)),
            mem_tag=torch.zeros((B, 1, 8, 2, 32)),
            mem_last=torch.full((B, 1, 8, 2), -1, dtype=torch.long),
        )

        routing = {"collect_aux": True}
        # Step 1: Routing
        routing.update(mem.compute_routing_step(u[:, :1], st, collect_aux=True))
        # Step 2: Write (this triggers the tuner update)
        _ = mem.write_chunk(u[:, :1], st, routing, 0, None)

        # Verify telemetry in routing
        self.assertIn("memory/utilization", routing)
        self.assertIn("memory/resonant/final_sim", routing)
        self.assertIn("tuner/resonant_coupling_mult", routing)

        # Verify that tuner mult is a float
        self.assertIsInstance(routing["tuner/resonant_coupling_mult"], float)


if __name__ == "__main__":
    unittest.main()
