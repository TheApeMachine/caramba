"""Fast train path for MOSAIC blocks"""

from typing import Any
import torch
from torch import Tensor, nn

from caramba.layer.memory_block.state_bank import StateBank
from caramba.layer.memory_block.memory import MemoryBlockMemory
from caramba.layer.memory_block.block.path import Path
from caramba.layer.memory_block.state import MemoryBlockState
from caramba.layer.memory_block.isa import MemoryOpcode


class FastTrainPath(Path):
    """Chunked training path

    Chunking keeps the semantics equivalent while reducing Python overhead,
    which matters when the block does a lot of “small” stateful work per token.
    """
    def __init__(
        self,
        *,
        state: StateBank,
        memory: MemoryBlockMemory,
        gate_long: nn.Linear,
        gate_mem: nn.Linear,
        chunk_size: int,
    ) -> None:
        super().__init__(
            state=state,
            memory=memory,
            gate_long=gate_long,
            gate_mem=gate_mem,
            chunk_size=chunk_size,
        )

    def forward(self, *args: Tensor, **kwargs: Tensor) -> Tensor:
        """Forward is not used; prefer `run(...)`.

        This exists to satisfy the abstract `Path.forward` contract so the class is
        instantiable. MOSAIC calls the explicit `run(...)` API.
        """
        _ = args
        if kwargs:
            # Best-effort support: if the caller passes the `run(...)` kwargs, return delta.
            delta, _out = self.run(  # type: ignore[arg-type]
                u=kwargs["u"],
                local=kwargs["local"],
                st=kwargs["st"],  # type: ignore[typeddict-item]
                routing=kwargs["routing"],  # type: ignore[typeddict-item]
                write_mask=kwargs.get("write_mask", None),
                opcode_ctrl=kwargs.get("opcode_ctrl", None),
            )
            return delta
        raise RuntimeError("FastTrainPath.forward is not used; call FastTrainPath.run(...) instead.")

    def run(
        self,
        *,
        u: Tensor,
        local: Tensor,
        st: MemoryBlockState,
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
            parts["r"].append(
                self.read_with_opcode(
                    u=u_c,
                    st=st,
                    routing=routing_c,
                    opcode_ctrl=opcode_ctrl,
                    t0=t0,
                    t1=t1,
                )
            )
            parts["gate"].append(
                self.write_chunk(
                    u=u_c,
                    st=st,
                    routing=routing_c,
                    write_mask=write_mask,
                    opcode_ctrl=opcode_ctrl,
                    t0=t0,
                    t1=t1,
                )
            )

        st.s = s.detach()
        st.step += int(u.size(1))

        return self.finalize(u=u, local=local, parts=parts)

    def prepare_parts(self, _st: MemoryBlockState) -> dict[str, list[Tensor]]:
        # `_st` is intentionally unused (reserved for future state-driven partitioning).
        return {"g": [], "r": [], "gate": [], "util": []}

    def chunks(self, *, u: Tensor, routing: dict[str, Any]):
        T = int(u.size(1))
        for t0 in range(0, T, int(self.chunk_size)):
            t1 = min(T, t0 + int(self.chunk_size))
            u_c = u[:, t0:t1, :]
            routing_c: dict[str, Any] = {}
            for k, v in routing.items():
                if isinstance(v, Tensor) and v.ndim >= 2 and int(v.size(1)) == int(T):
                    routing_c[k] = v[:, t0:t1]
                else:
                    routing_c[k] = v
            yield t0, t1, u_c, routing_c

    def write_chunk(
        self,
        *,
        u: Tensor,
        st: MemoryBlockState,
        routing: dict[str, Any],
        write_mask: Tensor | None,
        opcode_ctrl: Tensor | None,
        t0: int,
        t1: int,
    ) -> Tensor:
        mask_c = write_mask[:, t0:t1] if isinstance(write_mask, Tensor) else None
        scale = self.write_scale(opcode_ctrl=opcode_ctrl, t0=t0, t1=t1)
        return self.memory.write_chunk(
            u, st, routing, t0, mask_c, write_scale=scale,
        )

    def finalize(
        self,
        *,
        u: Tensor,
        local: Tensor,
        parts: dict[str, list[Tensor]],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        g_seq = torch.cat(parts["g"], dim=1)
        r_seq = torch.cat(parts["r"], dim=1)

        delta = local + torch.sigmoid(
            self.gate_long(u),
        ) * g_seq + torch.sigmoid(self.gate_mem(u)) * r_seq

        out = {
            "gate_logits": torch.cat(parts["gate"], dim=1),
            "util_logits": torch.cat(parts["util"], dim=1),
        }

        return delta, out

    def read_with_opcode(
        self,
        *,
        u: Tensor,
        st: MemoryBlockState,
        routing: dict[str, Any],
        opcode_ctrl: Tensor | None,
        t0: int,
        t1: int,
    ) -> Tensor:
        if isinstance(opcode_ctrl, Tensor) and int(MemoryOpcode.READ_MEM) < int(opcode_ctrl.size(-1)):
            rd = opcode_ctrl[:, t0:t1, int(MemoryOpcode.READ_MEM)]

            if not (rd > 0).any().item():
                return torch.zeros((
                    int(u.size(0)),
                    int(u.size(1)),
                    int(self.memory.mem_dim),
                ), device=u.device, dtype=u.dtype)

            return self.memory.read(u, st, routing) * rd.unsqueeze(-1)

        return self.memory.read(u, st, routing)

    def write_scale(self, *, opcode_ctrl: Tensor | None, t0: int, t1: int) -> Tensor | None:
        if isinstance(opcode_ctrl, Tensor) and int(MemoryOpcode.WRITE_MEM) < int(opcode_ctrl.size(-1)):
            return opcode_ctrl[:, t0:t1, int(MemoryOpcode.WRITE_MEM)]

        return None