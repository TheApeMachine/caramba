"""Sequential path for MOSAIC blocks"""

from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor, nn

from caramba.layer.mosaic.block.path import Path
from caramba.layer.mosaic.state import MosaicState
from caramba.layer.mosaic.memory import MosaicMemory
from caramba.layer.mosaic.state_bank import StateBank
from caramba.layer.mosaic.isa import MosaicOpcode


class SequentialPath(Path):
    """Exact streaming path

    This path updates state token-by-token, which is the reference behavior for
    decoding and is useful as a correctness baseline for faster paths.
    """
    def __init__(
        self,
        *,
        state: StateBank,
        memory: MosaicMemory,
        gate_long: nn.Linear,
        gate_mem: nn.Linear,
    ) -> None:
        super().__init__(
            state=state, memory=memory, gate_long=gate_long, gate_mem=gate_mem, chunk_size=1,
        )

    def forward(self, *args: Tensor, **kwargs: Tensor) -> Tensor:
        """Forward is not used; prefer `run(...)`.

        This exists to satisfy the abstract `Path.forward` contract so the class is
        instantiable. MOSAIC calls the explicit `run(...)` API.
        """
        _ = args
        if kwargs:
            delta, _out = self.run(  # type: ignore[arg-type]
                u=kwargs["u"],
                local=kwargs["local"],
                st=kwargs["st"],  # type: ignore[typeddict-item]
                routing=kwargs["routing"],  # type: ignore[typeddict-item]
                write_mask=kwargs.get("write_mask", None),
                opcode_ctrl=kwargs.get("opcode_ctrl", None),
            )
            return delta
        raise RuntimeError("SequentialPath.forward is not used; call SequentialPath.run(...) instead.")

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
        route_parts: dict[str, list[Tensor]] = {}
        s = st.s

        for t, u1, routing_t in self.steps(u=u, routing=routing, st=st):
            g_t, s = self.state.step(u1.squeeze(1), s=s)
            parts["g"].append(g_t.unsqueeze(1))
            parts["util"].append(self.memory.mem_utility_head(u1).squeeze(-1))
            parts["r"].append(
                self.read_with_opcode(
                    u=u1, st=st, routing=routing_t, opcode_ctrl=opcode_ctrl, t=t,
                )
            )
            parts["gate"].append(
                self.write_step(
                    u=u1, st=st, routing=routing_t, write_mask=write_mask, opcode_ctrl=opcode_ctrl, t=t,
                )
            )
            self.memory.update_rmf(st, routing_t)
            self.collect_routing_aux(route_parts, routing_t)
            st.step += 1

        st.s = s.detach()
        self.finish_routing_aux(routing, route_parts)

        return self.finalize(u=u, local=local, parts=parts)

    def steps(self, *, u: Tensor, routing: dict[str, Any], st: MosaicState):
        T = int(u.size(1))
        for t in range(T):
            u1 = u[:, t : t + 1, :]
            if bool(getattr(self.memory, "rmf_enabled", False)) and getattr(self.memory, "rmf", None) is not None:
                collect_aux = bool(routing.get("collect_aux", False))
                routing_t = self.memory.compute_routing_step(u1, st, collect_aux=collect_aux)
                routing_t["collect_aux"] = collect_aux
            else:
                # Only slice tensors that actually carry a time dimension.
                routing_t: dict[str, Any] = {}
                for k, v in routing.items():
                    if isinstance(v, Tensor) and v.ndim >= 2 and int(v.size(1)) == int(T):
                        routing_t[k] = v[:, t : t + 1]
                    else:
                        routing_t[k] = v
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
            if not (rd > 0).any().item():
                return torch.zeros((int(u.size(0)), 1, int(self.memory.mem_dim)), device=u.device, dtype=u.dtype)
            return self.memory.read(u, st, routing) * rd.view(int(u.size(0)), 1, 1)
        return self.memory.read(u, st, routing)

    def write_scale(self, *, opcode_ctrl: Tensor | None, t: int, B: int) -> Tensor | None:
        if isinstance(opcode_ctrl, Tensor) and int(MosaicOpcode.WRITE_MEM) < int(opcode_ctrl.size(-1)):
            ws = opcode_ctrl[:, t, int(MosaicOpcode.WRITE_MEM)]
            return ws.view(int(B), 1)

        return None

    def collect_routing_aux(self, route_parts: dict[str, list[Tensor]], routing_t: dict[str, Any]) -> None:
        if not bool(routing_t.get("collect_aux", False)):
            return

        for k in ("read_bit_logits", "write_bit_logits", "read_vq_logits", "write_vq_logits", "rmf_delta_rms", "rmf_field_rms"):
            v = routing_t.get(k, None)
            if isinstance(v, Tensor) and v.ndim >= 2:
                route_parts.setdefault(str(k), []).append(v)

    def finish_routing_aux(self, routing: dict[str, Any], route_parts: dict[str, list[Tensor]]) -> None:
        if not bool(routing.get("collect_aux", False)):
            return

        for k, parts in route_parts.items():
            if len(parts) > 0:
                routing[str(k)] = torch.cat(parts, dim=1)

