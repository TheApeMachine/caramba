"""
Unit tests for the platform-grade weight nowcaster.
"""

from __future__ import annotations

import unittest

import torch
from torch import nn

from orchestrator.nowcast import NowcastConfig, WeightNowcaster


class TestWeightNowcaster(unittest.TestCase):
    def test_smoke_record_and_nowcast_cpu(self) -> None:
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        cfg = NowcastConfig(
            horizon=3,
            nowcast_interval=2,
            min_steps_before_start=0,
            history_size=4,
            train_predictor_every=1,
            use_sketch=False,  # keep deterministic on CPU
            optimizer_state_policy="keep",
            advance_optimizer_steps_on_nowcast=True,
        )
        now = WeightNowcaster(model, optimizer=opt, config=cfg)

        did_nowcast = False
        last_step_state = None

        for step in range(6):
            opt.zero_grad(set_to_none=True)
            x = torch.randn(2, 8)
            y = torch.randn(2, 4)
            loss = (model(x) - y).pow(2).mean()
            loss.backward()
            opt.step()

            # record after update, before clearing grads
            now.record(step)

            if now.should_nowcast(step):
                # capture current optimizer step counter (AdamW)
                p0 = next(model.parameters())
                st = opt.state.get(p0, {})
                last_step_state = st.get("step")

                skipped = now.nowcast(step)
                self.assertEqual(skipped, 3)
                did_nowcast = True
                break

        self.assertTrue(did_nowcast)
        stats = now.get_stats()
        self.assertIn("num_nodes_tracked", stats)
        self.assertGreater(int(stats["num_nodes_tracked"]), 0)

        # Optimizer step should have advanced by horizon.
        p0 = next(model.parameters())
        st2 = opt.state.get(p0, {})
        step2 = st2.get("step")
        if isinstance(last_step_state, int) and isinstance(step2, int):
            self.assertEqual(step2, last_step_state + cfg.horizon)
        elif torch.is_tensor(last_step_state) and torch.is_tensor(step2):
            self.assertTrue(torch.allclose(step2, last_step_state + torch.as_tensor(cfg.horizon, dtype=step2.dtype)))

    def test_block_node_mode_increases_nodes(self) -> None:
        model = nn.Linear(8, 9, bias=False)
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)

        cfg = NowcastConfig(
            block_node_mode="out_channels",
            block_min_dim0=1,
            block_size=2,
            use_sketch=False,
            min_steps_before_start=0,
            nowcast_interval=1000000,
        )
        now = WeightNowcaster(model, optimizer=opt, config=cfg)
        stats = now.get_stats()

        # 9 out_channels with block_size=2 => 5 blocks => >= 5 nodes
        self.assertGreaterEqual(int(stats["num_nodes_tracked"]), 5)


if __name__ == "__main__":
    unittest.main()

