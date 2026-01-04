"""Tests for MOSAIC opcode control surface."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from caramba.config.layer import MosaicBlockLayerConfig
from caramba.layer.mosaic.block import MosaicBlockLayer
from caramba.layer.mosaic.isa import MosaicOpcode
from caramba.layer.mosaic.state import set_state


@dataclass(slots=True)
class _Ctx:
    mosaic_collect_aux: bool = False
    mosaic_stats_enabled: bool = False
    mosaic_teacher_p: float = 1.0
    mosaic_teacher: dict[str, torch.Tensor] | None = None


def _make_layer(*, d_model: int = 8) -> MosaicBlockLayer:
    cfg = MosaicBlockLayerConfig(
        d_model=d_model,
        conv_kernel=3,
        mlp_mult=1.0,
        dropout_p=0.0,
        state_k=2,
        state_decay_min=0.9,
        state_decay_max=0.9,
        mem_router="bits",
        mem_buckets=8,
        mem_dim=d_model,
        mem_hashes=1,
        mem_assoc=2,
        mem_key_dim=4,
        mem_read_temp=1.0,
        mem_write_threshold=0.99,  # effectively disable writes
        mem_write_eta=0.0,
        forced_read_dropout_p=0.0,
        opcodes_enabled=True,
        opcode_vocab=4,
        opcodes_control_enabled=True,
        opcodes_control_temp=1.0,
        gate_long_init=-10.0,  # suppress state bank contribution
        gate_mem_init=10.0,  # amplify memory contribution
    )
    layer = MosaicBlockLayer(cfg)

    # Zero out local mixer and state bank so output is dominated by memory read.
    with torch.no_grad():
        # Local mixer is now a composed module.
        layer.local_mixer.conv.weight.zero_()
        layer.local_mixer.gate_proj.weight.zero_()
        layer.local_mixer.gate_proj.bias.zero_()
        layer.local_mixer.mlp_up.weight.zero_()
        layer.local_mixer.mlp_up.bias.zero_()
        layer.local_mixer.mlp_down.weight.zero_()
        layer.local_mixer.mlp_down.bias.zero_()

        # State bank is now a composed module.
        layer.state_bank.state_in.weight.zero_()
        layer.state_bank.state_out.weight.zero_()

        # Identity mem_out (mem_dim == d_model).
        layer.memory.mem_out.weight.copy_(torch.eye(d_model))

        # Make opcode logits depend only on bias (u is near-constant in this test).
        assert layer.opcode_head is not None
        layer.opcode_head.weight.zero_()

        # Make fusion gates depend only on bias.
        layer.gate_long.weight.zero_()
        layer.gate_mem.weight.zero_()

    return layer


def _prefill_memory(layer: MosaicBlockLayer, ctx: _Ctx, *, B: int, device: torch.device, dtype: torch.dtype) -> None:
    st = layer.init_state(B, device, dtype)

    # Mark slot 0 as valid and fill it with a non-zero vector.
    st.mem_last.fill_(-1)
    st.mem_last[:, :, 0, 0] = 0
    st.mem_v.zero_()
    st.mem_v[:, :, 0, 0, :] = 1.0

    set_state(ctx, layer.ctx_key, st)


def test_opcode_control_gates_memory_read_output() -> None:
    torch.manual_seed(0)
    layer = _make_layer(d_model=8)
    layer.train()

    B, T, D = 1, 2, 8
    x = torch.zeros((B, T, D), dtype=torch.float32)

    ctx = _Ctx(
        mosaic_collect_aux=False,
        mosaic_stats_enabled=False,
        mosaic_teacher_p=1.0,
        mosaic_teacher={
            "read_bucket": torch.zeros((B, T, layer.memory.mem_hashes), dtype=torch.long),
        },
    )

    _prefill_memory(layer, ctx, B=B, device=x.device, dtype=x.dtype)

    # Case 1: force opcode = READ -> output should include memory contribution.
    with torch.no_grad():
        assert layer.opcode_head is not None
        layer.opcode_head.bias.zero_()
        layer.opcode_head.bias[int(MosaicOpcode.READ_MEM)] = 10.0

    y_read = layer(x, ctx=ctx)
    assert torch.isfinite(y_read).all()
    assert float(y_read.abs().sum().item()) > 0.0

    # Case 2: force opcode = NOP -> memory read skipped -> output near zero.
    with torch.no_grad():
        layer.opcode_head.bias.zero_()
        layer.opcode_head.bias[int(MosaicOpcode.NOP)] = 10.0

    y_nop = layer(x, ctx=ctx)
    assert torch.isfinite(y_nop).all()
    assert float(y_nop.abs().sum().item()) == 0.0


def test_opcode_control_uses_ste_gradients() -> None:
    torch.manual_seed(0)
    layer = _make_layer(d_model=8)
    layer.train()

    B, T, D = 1, 2, 8
    x = torch.zeros((B, T, D), dtype=torch.float32)

    ctx = _Ctx(
        mosaic_collect_aux=False,
        mosaic_stats_enabled=False,
        mosaic_teacher_p=1.0,
        mosaic_teacher={
            "read_bucket": torch.zeros((B, T, layer.memory.mem_hashes), dtype=torch.long),
        },
    )

    _prefill_memory(layer, ctx, B=B, device=x.device, dtype=x.dtype)

    # Force opcode = READ so the controlled read path is active and contributes to loss.
    with torch.no_grad():
        assert layer.opcode_head is not None
        layer.opcode_head.bias.zero_()
        layer.opcode_head.bias[int(MosaicOpcode.READ_MEM)] = 10.0

    y = layer(x, ctx=ctx)
    loss = y.sum()
    loss.backward()

    assert layer.opcode_head is not None
    assert layer.opcode_head.bias.grad is not None
    assert float(layer.opcode_head.bias.grad.abs().sum().item()) > 0.0

