"""Tests for the MOSAIC block layer."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from caramba.config.layer import LayerType, MosaicBlockLayerConfig
from caramba.layer.mosaic_block import MosaicBlockLayer


class MosaicBlockLayerTest(unittest.TestCase):
    def test_forward_shape(self) -> None:
        torch.manual_seed(0)
        cfg = MosaicBlockLayerConfig(
            type=LayerType.MOSAIC_BLOCK,
            d_model=32,
            conv_kernel=5,
            mlp_mult=2.0,
            dropout_p=0.0,
            state_k=4,
            mem_buckets=256,
            mem_dim=16,
            mem_hashes=2,
            mem_write_threshold=0.99,  # mostly off
            mem_write_eta=0.1,
        )
        layer = cfg.build()
        assert isinstance(layer, MosaicBlockLayer)
        x = torch.randn(2, 7, 32)
        y = layer(x, ctx=None)  # type: ignore[call-arg]
        self.assertEqual(tuple(y.shape), (2, 7, 32))

    def test_streaming_matches_full(self) -> None:
        torch.manual_seed(0)
        cfg = MosaicBlockLayerConfig(
            type=LayerType.MOSAIC_BLOCK,
            d_model=16,
            conv_kernel=7,
            mlp_mult=1.5,
            dropout_p=0.0,
            state_k=3,
            state_decay_min=0.9,
            state_decay_max=0.99,
            mem_buckets=128,
            mem_dim=8,
            mem_hashes=2,
            mem_write_threshold=0.5,
            mem_write_eta=0.2,
        )
        layer = cfg.build()
        assert isinstance(layer, MosaicBlockLayer)
        layer.eval()

        B, T, D = 1, 12, 16
        x = torch.randn(B, T, D)

        # Full sequence.
        y_full = layer(x, ctx=None)  # type: ignore[call-arg]

        # Streaming one token at a time with persistent ctx.
        ctx = SimpleNamespace()
        ys = []
        for t in range(T):
            yt = layer(x[:, t : t + 1, :], ctx=ctx)  # type: ignore[call-arg]
            ys.append(yt)
        y_stream = torch.cat(ys, dim=1)

        self.assertTrue(torch.allclose(y_full, y_stream, atol=1e-5, rtol=1e-5))

    def test_streaming_matches_full_with_registers(self) -> None:
        torch.manual_seed(0)
        cfg = MosaicBlockLayerConfig(
            type=LayerType.MOSAIC_BLOCK,
            d_model=16,
            conv_kernel=5,
            mlp_mult=1.5,
            dropout_p=0.0,
            state_k=3,
            state_decay_min=0.9,
            state_decay_max=0.99,
            mem_buckets=128,
            mem_dim=8,
            mem_hashes=2,
            mem_write_threshold=0.5,
            mem_write_eta=0.2,
            reg_slots=4,
            reg_write_threshold=0.5,
            reg_write_eta=1.0,
        )
        layer = cfg.build()
        assert isinstance(layer, MosaicBlockLayer)
        layer.eval()

        B, T, D = 1, 12, 16
        x = torch.randn(B, T, D)

        y_full = layer(x, ctx=None)  # type: ignore[call-arg]

        ctx = SimpleNamespace()
        ys = []
        for t in range(T):
            yt = layer(x[:, t : t + 1, :], ctx=ctx)  # type: ignore[call-arg]
            ys.append(yt)
        y_stream = torch.cat(ys, dim=1)

        self.assertTrue(torch.allclose(y_full, y_stream, atol=1e-5, rtol=1e-5))

    def test_registers_last_write_wins(self) -> None:
        torch.manual_seed(0)
        cfg = MosaicBlockLayerConfig(
            type=LayerType.MOSAIC_BLOCK,
            d_model=8,
            conv_kernel=3,
            mlp_mult=1.0,
            dropout_p=0.0,
            state_k=1,
            state_decay_min=0.99,
            state_decay_max=0.99,
            mem_buckets=64,
            mem_dim=8,
            mem_hashes=1,
            mem_write_threshold=0.99,  # keep hash memory mostly off
            mem_write_eta=0.1,
            reg_slots=2,
            reg_write_threshold=0.0,  # we will force the gate on via bias
            reg_write_eta=1.0,
        )
        layer = cfg.build()
        assert isinstance(layer, MosaicBlockLayer)
        layer.eval()

        # Force deterministic register behavior:
        # - always write
        # - always choose slot 0
        # - write value = identity(ut)
        assert layer.reg_write_gate is not None
        assert layer.reg_sel is not None
        assert layer.reg_value is not None
        with torch.no_grad():
            layer.reg_write_gate.weight.zero_()
            layer.reg_write_gate.bias.fill_(20.0)  # sigmoid ~ 1

            layer.reg_sel.weight.zero_()
            layer.reg_sel.bias.fill_(0.0)
            layer.reg_sel.bias[0] = 10.0  # pick slot 0

            layer.reg_value.weight.zero_()
            layer.reg_value.bias.zero_()
            layer.reg_value.weight.copy_(torch.eye(8))

        B, T, D = 1, 3, 8
        x = torch.randn(B, T, D)
        # Enable aux collection to retrieve regs.
        ctx = SimpleNamespace(mosaic_collect_aux=True)
        _ = layer(x, ctx=ctx)  # type: ignore[call-arg]
        aux = getattr(ctx, "mosaic_aux_out", {})
        self.assertIn("mosaic_regs_last", aux)

        regs_last = aux["mosaic_regs_last"]
        self.assertEqual(tuple(regs_last.shape), (B, 2, D))

        # The final value in slot 0 should match the final pre-norm vector u_t.
        # (reg_value is identity and we overwrite each step).
        u_last = x[:, -1, :]
        u_last = u_last * torch.rsqrt(u_last.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        self.assertTrue(torch.allclose(regs_last[:, 0, :], u_last, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(regs_last[:, 1, :], torch.zeros_like(regs_last[:, 1, :])))


if __name__ == "__main__":
    unittest.main()

