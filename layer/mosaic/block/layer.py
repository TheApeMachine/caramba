"""MOSAIC block layer

Implements a streaming, shape-preserving block:
- local mixer (causal conv + gated MLP)
- multiscale state bank (leaky integrators)
- hard-addressed memory (tag-routed, set-associative)

This is an explicit-memory alternative to attention/KV caches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from caramba.config.layer import MosaicBlockLayerConfig
from caramba.layer.mosaic.isa import MosaicOpcode
from caramba.layer.mosaic.memory import MosaicMemory
from caramba.layer.mosaic.state import MosaicState, get_state, set_state
from caramba.layer.mosaic.block.local_mixer import LocalMixer
from caramba.layer.mosaic.block.norm import RmsNorm
from caramba.layer.mosaic.block.paths import FastTrainPath, SequentialPath
from caramba.layer.mosaic.block.state_bank import StateBank


@dataclass(frozen=True, slots=True)
class OpcodeControl:
    """Opcode control surface.

    Uses straight-through selection (hard forward, soft backward) for gating.
    """

    vocab: int
    temp: float

    def control(self, logits: Tensor, *, dtype: torch.dtype) -> Tensor:
        if logits.ndim != 3:
            raise ValueError(f"opcode logits must have shape (B,T,V), got {tuple(logits.shape)}")
        if float(self.temp) <= 0.0:
            raise ValueError("OpcodeControl.temp must be > 0")
        soft = torch.softmax((logits.float() / float(self.temp)), dim=-1)
        idx = soft.argmax(dim=-1)
        hard = torch.nn.functional.one_hot(idx, num_classes=int(self.vocab)).to(dtype=soft.dtype)
        sel = (hard - soft).detach() + soft
        return sel.to(dtype=dtype)


class MosaicBlockLayer(nn.Module):
    """MOSAIC block layer."""

    def __init__(self, config: MosaicBlockLayerConfig) -> None:
        super().__init__()
        self.config = config
        self.ctx_key = f"mosaic_block::{id(self)}"
        self.norm = RmsNorm(eps=1e-6)
        self.d_model = int(config.d_model)

        self.local_mixer = self.build_local_mixer()
        self.state_bank = self.build_state_bank()
        self.memory = MosaicMemory(config, int(self.d_model))

        self.gate_long = nn.Linear(int(self.d_model), 1, bias=True)
        self.gate_mem = nn.Linear(int(self.d_model), 1, bias=True)
        with torch.no_grad():
            self.gate_long.bias.fill_(float(getattr(config, "gate_long_init", 0.0)))
            self.gate_mem.bias.fill_(float(getattr(config, "gate_mem_init", 0.0)))

        self.opcodes_enabled = bool(getattr(config, "opcodes_enabled", False))
        self.opcodes_control_enabled = bool(getattr(config, "opcodes_control_enabled", False))
        self.opcode_vocab = int(getattr(config, "opcode_vocab", 4))
        self.opcodes_control_temp = float(getattr(config, "opcodes_control_temp", 1.0))
        self.opcode_head: nn.Linear | None = None
        if self.opcodes_enabled:
            self.opcode_head = nn.Linear(int(self.d_model), int(self.opcode_vocab), bias=True)

        self.fast_path = FastTrainPath(
            state=self.state_bank,
            memory=self.memory,
            gate_long=self.gate_long,
            gate_mem=self.gate_mem,
            chunk_size=int(getattr(config, "mem_train_chunk_size", 128)),
        )
        self.seq_path = SequentialPath(
            state=self.state_bank,
            memory=self.memory,
            gate_long=self.gate_long,
            gate_mem=self.gate_mem,
        )

    def build_local_mixer(self) -> LocalMixer:
        d = int(self.d_model)
        k = int(getattr(self.config, "conv_kernel", 7))
        conv = nn.Conv1d(d, d, kernel_size=k, padding=k - 1, groups=d, bias=False)
        gate_proj = nn.Linear(d, d, bias=True)
        hidden = max(1, int(round(float(getattr(self.config, "mlp_mult", 2.0)) * d)))
        mlp_up = nn.Linear(d, hidden, bias=True)
        mlp_down = nn.Linear(hidden, d, bias=True)
        dropout = nn.Dropout(float(getattr(self.config, "dropout_p", 0.0)))
        return LocalMixer(conv=conv, gate_proj=gate_proj, mlp_up=mlp_up, mlp_down=mlp_down, dropout=dropout, conv_kernel=k)

    def build_state_bank(self) -> StateBank:
        d = int(self.d_model)
        K = int(getattr(self.config, "state_k", 16))
        state_in = nn.Linear(d, K * d, bias=False)
        state_out = nn.Linear(K * d, d, bias=False)
        dmin = float(getattr(self.config, "state_decay_min", 0.90))
        dmax = float(getattr(self.config, "state_decay_max", 0.999))
        if abs(dmin - dmax) < 1e-12:
            decays = torch.full((K,), float(dmin))
        else:
            lo = max(1e-4, min(1.0 - 1e-4, dmin))
            hi = max(1e-4, min(1.0 - 1e-4, dmax))
            decays = torch.exp(torch.linspace(math.log(lo), math.log(hi), K))
        init = torch.log(decays) - torch.log1p(-decays)
        decay_logit = nn.Parameter(init)
        return StateBank(state_k=K, state_in=state_in, state_out=state_out, decay_logit=decay_logit)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> MosaicState:
        B = int(batch_size)
        D = int(self.d_model)
        k = int(getattr(self.config, "conv_kernel", 7))
        conv_buf = torch.zeros((B, max(0, k - 1), D), device=device, dtype=dtype)
        s = torch.zeros((B, int(self.state_bank.state_k), D), device=device, dtype=dtype)
        mem_k = torch.zeros((B, self.memory.mem_hashes, self.memory.mem_buckets, self.memory.mem_assoc, self.memory.mem_key_dim), device=device, dtype=dtype)
        mem_v = torch.zeros((B, self.memory.mem_hashes, self.memory.mem_buckets, self.memory.mem_assoc, self.memory.mem_dim), device=device, dtype=dtype)
        mem_last = torch.full((B, self.memory.mem_hashes, self.memory.mem_buckets, self.memory.mem_assoc), -1, device=device, dtype=torch.long)
        return MosaicState(conv_buf=conv_buf, s=s, regs=None, step=0, mem_k=mem_k, mem_v=mem_v, mem_last=mem_last)

    def forward(self, x: Tensor, *, ctx: Any | None = None) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B,T,D), got {tuple(x.shape)}")
        B, T, D = x.shape
        if int(D) != int(self.d_model):
            raise ValueError(f"Expected d_model={int(self.d_model)}, got {int(D)}")
        st = get_state(ctx, self.ctx_key)
        if st is None or int(st.s.size(0)) != int(B):
            st = self.init_state(int(B), x.device, x.dtype)

        u = self.norm.apply(x)
        local, new_buf = self.local_mixer.forward(u, state=st)
        if new_buf is not None:
            st.conv_buf = new_buf

        collect_aux = bool(getattr(ctx, "mosaic_collect_aux", False)) if ctx is not None else False
        routing = self.memory.compute_routing(u, collect_aux=collect_aux)
        routing["collect_aux"] = bool(collect_aux)
        write_mask = self.resolve_write_mask(ctx, B=B, T=T, device=x.device)
        opcode_ctrl = self.compute_opcode_control(u, collect_aux=collect_aux)

        use_fast = bool(self.training) and int(T) > 1 and not bool(getattr(ctx, "mosaic_stats_enabled", False))
        if use_fast:
            delta, outputs = self.fast_path.run(u=u, local=local, st=st, routing=routing, write_mask=write_mask, opcode_ctrl=opcode_ctrl)
        else:
            delta, outputs = self.seq_path.run(u=u, local=local, st=st, routing=routing, write_mask=write_mask, opcode_ctrl=opcode_ctrl)

        y = x + delta
        set_state(ctx, self.ctx_key, st)
        if collect_aux and ctx is not None:
            self.save_aux(ctx, outputs=outputs, routing=routing, opcode_logits=self.get_opcode_logits(u))
        return y

    def resolve_write_mask(self, ctx: Any | None, *, B: int, T: int, device: torch.device) -> Tensor | None:
        teacher = getattr(ctx, "mosaic_teacher", None) if ctx is not None else None
        if not (isinstance(teacher, dict) and "write_gate" in teacher):
            return None
        wg = teacher["write_gate"]
        if not isinstance(wg, Tensor) or tuple(wg.shape) != (int(B), int(T)):
            raise ValueError(f"teacher['write_gate'] must have shape (B,T)={(B,T)}, got {getattr(wg, 'shape', None)}")
        return wg.to(device=device, dtype=torch.float32)

    def compute_opcode_control(self, u: Tensor, *, collect_aux: bool) -> Tensor | None:
        if not bool(self.opcodes_enabled) or self.opcode_head is None:
            return None
        if not bool(self.opcodes_control_enabled):
            return None
        ctrl = OpcodeControl(vocab=int(self.opcode_vocab), temp=float(self.opcodes_control_temp))
        return ctrl.control(self.opcode_head(u), dtype=u.dtype)

    def get_opcode_logits(self, u: Tensor) -> Tensor | None:
        if not bool(self.opcodes_enabled) or self.opcode_head is None:
            return None
        return self.opcode_head(u)

    class _AuxCtx(Protocol):
        mosaic_aux_out: dict[str, Tensor] | None

    def save_aux(
        self,
        ctx: _AuxCtx,
        *,
        outputs: dict[str, Tensor],
        routing: dict[str, Any],
        opcode_logits: Tensor | None,
    ) -> None:
        aux = getattr(ctx, "mosaic_aux_out", None)
        if aux is None:
            aux = {}
        if not isinstance(aux, dict):
            raise TypeError("ctx.mosaic_aux_out must be a dict when collecting aux")
        aux["mosaic_write_gate_logits"] = outputs["gate_logits"]
        aux["mosaic_write_utility_logits"] = outputs["util_logits"]
        if isinstance(opcode_logits, Tensor):
            aux["mosaic_opcode_logits"] = opcode_logits
        if "read_bit_logits" in routing:
            aux["mosaic_read_bit_logits"] = routing["read_bit_logits"]
        if "write_bit_logits" in routing:
            aux["mosaic_write_bit_logits"] = routing["write_bit_logits"]
        if "read_vq_logits" in routing:
            aux["mosaic_vq_read_logits"] = routing["read_vq_logits"]
        if "write_vq_logits" in routing:
            aux["mosaic_vq_write_logits"] = routing["write_vq_logits"]
        ctx.mosaic_aux_out = aux

