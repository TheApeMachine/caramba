"""MOSAIC forward paths

Contains the two compute paths for MOSAIC blocks:
- FastTrainPath: chunked training path (T>1) to reduce overhead
- SequentialPath: exact streaming semantics path
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from caramba.layer.mosaic.isa import MosaicOpcode
from caramba.layer.mosaic.memory import MosaicMemory
from caramba.layer.mosaic.state import MosaicState
from caramba.layer.mosaic.block.state_bank import StateBank


@dataclass(slots=True)
class FastTrainPath:
    """Chunked training path."""

    state: StateBank
    memory: MosaicMemory
    gate_long: nn.Linear
    gate_mem: nn.Linear
    chunk_size: int

    def run(
        self,
        *,
        u: Tensor,
        local: Tensor,
        st: MosaicState,
        routing: dict[str, Any],
        write_mask: Tensor | None,
        opcode_ctrl: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        parts = self.prepare_parts(st)
        s = st.s
        for t0, t1, u_c, routing_c in self.chunks(u=u, routing=routing):
            g_c, s = self.state.scan(u_c, s0=s)
            parts["g"].append(g_c)
            parts["util"].append(self.memory.mem_utility_head(u_c).squeeze(-1))
            parts["r"].append(self.read_with_opcode(u=u_c, st=st, routing=routing_c, opcode_ctrl=opcode_ctrl, t0=t0, t1=t1))
            parts["gate"].append(self.write_chunk(u=u_c, st=st, routing=routing_c, write_mask=write_mask, opcode_ctrl=opcode_ctrl, t0=t0, t1=t1))
        st.s = s.detach()
        st.step += int(u.size(1))
        return self.finalize(u=u, local=local, parts=parts)

    def prepare_parts(self, st: MosaicState) -> dict[str, list[Tensor]]:
        return {"g": [], "r": [], "gate": [], "util": []}

    def chunks(self, *, u: Tensor, routing: dict[str, Any]):
        T = int(u.size(1))
        for t0 in range(0, T, int(self.chunk_size)):
            t1 = min(T, t0 + int(self.chunk_size))
            u_c = u[:, t0:t1, :]
            routing_c = {k: v[:, t0:t1] if isinstance(v, Tensor) else v for k, v in routing.items()}
            yield t0, t1, u_c, routing_c

    def write_chunk(
        self,
        *,
        u: Tensor,
        st: MosaicState,
        routing: dict[str, Any],
        write_mask: Tensor | None,
        opcode_ctrl: Tensor | None,
        t0: int,
        t1: int,
    ) -> Tensor:
        mask_c = write_mask[:, t0:t1] if isinstance(write_mask, Tensor) else None
        scale = self.write_scale(opcode_ctrl=opcode_ctrl, t0=t0, t1=t1)
        return self.memory.write_chunk(u, st, routing, t0, mask_c, write_scale=scale)

    def finalize(self, *, u: Tensor, local: Tensor, parts: dict[str, list[Tensor]]) -> tuple[Tensor, dict[str, Tensor]]:
        g_seq = torch.cat(parts["g"], dim=1)
        r_seq = torch.cat(parts["r"], dim=1)
        delta = local + torch.sigmoid(self.gate_long(u)) * g_seq + torch.sigmoid(self.gate_mem(u)) * r_seq
        out = {"gate_logits": torch.cat(parts["gate"], dim=1), "util_logits": torch.cat(parts["util"], dim=1)}
        return delta, out

    def read_with_opcode(
        self,
        *,
        u: Tensor,
        st: MosaicState,
        routing: dict[str, Any],
        opcode_ctrl: Tensor | None,
        t0: int,
        t1: int,
    ) -> Tensor:
        if isinstance(opcode_ctrl, Tensor) and int(MosaicOpcode.READ_MEM) < int(opcode_ctrl.size(-1)):
            rd = opcode_ctrl[:, t0:t1, int(MosaicOpcode.READ_MEM)]
            if not bool((rd > 0).any()):
                return torch.zeros((int(u.size(0)), int(u.size(1)), int(self.memory.mem_dim)), device=u.device, dtype=u.dtype)
            return self.memory.read(u, st, routing) * rd.unsqueeze(-1)
        return self.memory.read(u, st, routing)

    def write_scale(self, *, opcode_ctrl: Tensor | None, t0: int, t1: int) -> Tensor | None:
        if isinstance(opcode_ctrl, Tensor) and int(MosaicOpcode.WRITE_MEM) < int(opcode_ctrl.size(-1)):
            return opcode_ctrl[:, t0:t1, int(MosaicOpcode.WRITE_MEM)]
        return None


@dataclass(slots=True)
class SequentialPath:
    """Exact streaming path."""

    state: StateBank
    memory: MosaicMemory
    gate_long: nn.Linear
    gate_mem: nn.Linear

    def run(
        self,
        *,
        u: Tensor,
        local: Tensor,
        st: MosaicState,
        routing: dict[str, Any],
        write_mask: Tensor | None,
        opcode_ctrl: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        parts = {"g": [], "r": [], "gate": [], "util": []}
        s = st.s
        for t, u1, routing_t in self.steps(u=u, routing=routing):
            g_t, s = self.state.step(u1.squeeze(1), s=s)
            parts["g"].append(g_t.unsqueeze(1))
            parts["util"].append(self.memory.mem_utility_head(u1).squeeze(-1))
            parts["r"].append(self.read_with_opcode(u=u1, st=st, routing=routing_t, opcode_ctrl=opcode_ctrl, t=t))
            parts["gate"].append(self.write_step(u=u1, st=st, routing=routing_t, write_mask=write_mask, opcode_ctrl=opcode_ctrl, t=t))
            st.step += 1
        st.s = s.detach()
        return self.finalize(u=u, local=local, parts=parts)

    def steps(self, *, u: Tensor, routing: dict[str, Any]):
        T = int(u.size(1))
        for t in range(T):
            u1 = u[:, t : t + 1, :]
            routing_t = {k: v[:, t : t + 1] if isinstance(v, Tensor) else v for k, v in routing.items()}
            yield t, u1, routing_t

    def write_step(
        self,
        *,
        u: Tensor,
        st: MosaicState,
        routing: dict[str, Any],
        write_mask: Tensor | None,
        opcode_ctrl: Tensor | None,
        t: int,
    ) -> Tensor:
        B = int(u.size(0))
        mask_t = write_mask[:, t : t + 1] if isinstance(write_mask, Tensor) else None
        scale = self.write_scale(opcode_ctrl=opcode_ctrl, t=t, B=B)
        return self.memory.write_chunk(u, st, routing, 0, mask_t, write_scale=scale)

    def finalize(self, *, u: Tensor, local: Tensor, parts: dict[str, list[Tensor]]) -> tuple[Tensor, dict[str, Tensor]]:
        g_seq = torch.cat(parts["g"], dim=1)
        r_seq = torch.cat(parts["r"], dim=1)
        delta = local + torch.sigmoid(self.gate_long(u)) * g_seq + torch.sigmoid(self.gate_mem(u)) * r_seq
        out = {"gate_logits": torch.cat(parts["gate"], dim=1), "util_logits": torch.cat(parts["util"], dim=1)}
        return delta, out

    def read_with_opcode(self, *, u: Tensor, st: MosaicState, routing: dict[str, Any], opcode_ctrl: Tensor | None, t: int) -> Tensor:
        if isinstance(opcode_ctrl, Tensor) and int(MosaicOpcode.READ_MEM) < int(opcode_ctrl.size(-1)):
            rd = opcode_ctrl[:, t, int(MosaicOpcode.READ_MEM)]
            if not bool((rd > 0).any()):
                return torch.zeros((int(u.size(0)), 1, int(self.memory.mem_dim)), device=u.device, dtype=u.dtype)
            return self.memory.read(u, st, routing) * rd.view(int(u.size(0)), 1, 1)
        return self.memory.read(u, st, routing)

    def write_scale(self, *, opcode_ctrl: Tensor | None, t: int, B: int) -> Tensor | None:
        if isinstance(opcode_ctrl, Tensor) and int(MosaicOpcode.WRITE_MEM) < int(opcode_ctrl.size(-1)):
            ws = opcode_ctrl[:, t, int(MosaicOpcode.WRITE_MEM)]
            return ws.view(int(B), 1)
        return None

