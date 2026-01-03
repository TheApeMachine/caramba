"""MOSAIC block layer: no attention, no KV cache.

Implements a streaming, shape-preserving block that combines:
- Local mixer: depthwise causal conv + gated MLP
- Multiscale continuous state bank: leaky integrators across K timescales
- Hard-addressed associative cache: fixed-size hash table with O(1) read/write

This is an explicit-memory alternative to transformer attention/KV caches.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from caramba.carmath import leaky_integrator_scan
from caramba.config.layer import MosaicBlockLayerConfig
from caramba.layer.mosaic.isa import MosaicOpcode
from caramba.layer.mosaic.state import MosaicState, get_state, set_state
from caramba.layer.mosaic.memory import MosaicMemory


def _rms_norm(x: Tensor, *, eps: float = 1e-6) -> Tensor:
    # x: (..., d)
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + float(eps))


class MosaicBlockLayer(nn.Module):
    """Streaming MOSAIC block (shape preserving)."""

    def __init__(self, config: MosaicBlockLayerConfig) -> None:
        super().__init__()
        self.config = config

        d = int(config.d_model)
        k = int(config.conv_kernel)
        if k < 1:
            raise ValueError(f"conv_kernel must be >= 1, got {k}")

        # Local mixer.
        self.conv = nn.Conv1d(
            in_channels=d,
            out_channels=d,
            kernel_size=k,
            padding=k - 1,  # causal: we'll slice to length T
            groups=d,
            bias=False,
        )
        self.gate_proj = nn.Linear(d, d, bias=True)

        hidden = max(1, int(round(float(config.mlp_mult) * d)))
        self.mlp_up = nn.Linear(d, hidden, bias=True)
        self.mlp_down = nn.Linear(hidden, d, bias=True)
        self.dropout = nn.Dropout(float(config.dropout_p))

        # Multiscale state bank.
        K = int(config.state_k)
        if K < 1:
            raise ValueError(f"state_k must be >= 1, got {K}")
        self.state_k = K
        self.state_in = nn.Linear(d, K * d, bias=False)
        self.state_out = nn.Linear(K * d, d, bias=False)

        # Learnable decays
        dmin = float(config.state_decay_min)
        dmax = float(config.state_decay_max)
        if not (0.0 <= dmin <= 1.0 and 0.0 <= dmax <= 1.0 and dmin <= dmax):
            raise ValueError(f"Invalid state_decay range: min={dmin}, max={dmax}")

        rmin = float(getattr(config, "state_decay_reg_min", 0.001))
        rmax = float(getattr(config, "state_decay_reg_max", 0.999))
        if not (0.0 <= rmin <= 1.0 and 0.0 <= rmax <= 1.0 and rmin <= rmax):
            raise ValueError(f"Invalid state_decay_reg range: min={rmin}, max={rmax}")
        self.state_decay_reg_min = rmin
        self.state_decay_reg_max = rmax

        if dmin == 0.0 and dmax == 0.0:
            init = torch.zeros(K)
        else:
            lo = max(1e-4, min(1.0 - 1e-4, dmin))
            hi = max(1e-4, min(1.0 - 1e-4, dmax))
            if abs(lo - hi) < 1e-12:
                decays = torch.full((K,), float(lo))
            else:
                decays = torch.exp(torch.linspace(math.log(lo), math.log(hi), K))
            init = torch.log(decays) - torch.log1p(-decays)  # logit
        self.state_decay_logit = nn.Parameter(init)

        # Memory subsystem
        self.memory = MosaicMemory(config, d)

        # Registers
        reg_slots = getattr(config, "reg_slots", None)
        self.reg_slots = int(reg_slots) if reg_slots is not None else 0
        self.reg_write_gate: nn.Linear | None = None
        self.reg_sel: nn.Linear | None = None
        self.reg_value: nn.Linear | None = None
        self.gate_reg: nn.Linear | None = None
        if self.reg_slots > 0:
            self.reg_write_gate = nn.Linear(d, 1, bias=True)
            self.reg_sel = nn.Linear(d, self.reg_slots, bias=True)
            self.reg_value = nn.Linear(d, d, bias=True)
            self.gate_reg = nn.Linear(d, 1, bias=True)
            with torch.no_grad():
                self.gate_reg.bias.fill_(float(getattr(config, "gate_reg_init", 0.0)))

        # Opcodes
        self.opcodes_enabled = bool(getattr(config, "opcodes_enabled", False))
        self.opcode_vocab = int(getattr(config, "opcode_vocab", 4))
        self.opcodes_control_enabled = bool(getattr(config, "opcodes_control_enabled", False))
        self.opcodes_control_temp = float(getattr(config, "opcodes_control_temp", 1.0))
        self.opcode_head: nn.Linear | None = None
        if self.opcodes_enabled:
            if self.opcode_vocab < 2:
                raise ValueError(f"opcode_vocab must be >= 2, got {self.opcode_vocab}")
            self.opcode_head = nn.Linear(d, self.opcode_vocab, bias=True)
        if self.opcodes_control_enabled:
            if not self.opcodes_enabled:
                raise ValueError("opcodes_control_enabled requires opcodes_enabled=true")
            if self.opcodes_control_temp <= 0.0:
                raise ValueError(f"opcodes_control_temp must be > 0, got {self.opcodes_control_temp}")
            min_vocab = int(MosaicOpcode.WRITE_MEM) + 1
            if self.opcode_vocab < min_vocab:
                raise ValueError(
                    f"opcodes_control_enabled requires opcode_vocab >= {min_vocab} "
                    f"(to include NOP/READ/WRITE), got {self.opcode_vocab}"
                )

        # Fusion gates
        self.gate_long = nn.Linear(d, 1, bias=True)
        self.gate_mem = nn.Linear(d, 1, bias=True)
        with torch.no_grad():
            self.gate_long.bias.fill_(float(config.gate_long_init))
            self.gate_mem.bias.fill_(float(config.gate_mem_init))

        self._ctx_key = f"mosaic_block::{id(self)}"

    def _local_mixer(self, u: Tensor, *, ctx_state: MosaicState | None) -> tuple[Tensor, Tensor | None]:
        """Local mixer output + updated conv buffer (if streaming)."""
        B, T, D = u.shape
        k = int(self.config.conv_kernel)
        new_buf: Tensor | None = None

        if ctx_state is not None and int(T) == 1 and k > 1:
            window = torch.cat([ctx_state.conv_buf.to(dtype=u.dtype, device=u.device), u], dim=1)
            x = F.conv1d(
                window.transpose(1, 2),
                self.conv.weight,
                bias=None,
                stride=1,
                padding=0,
                dilation=1,
                groups=D,
            ).transpose(1, 2)
            new_buf = window[:, 1:, :].detach()
        else:
            x = self.conv(u.transpose(1, 2))[:, :, :T].transpose(1, 2)
            if ctx_state is not None and k > 1:
                keep = min(k - 1, int(T))
                new_buf = u[:, -keep:, :].detach() if keep > 0 else ctx_state.conv_buf

        gate = torch.sigmoid(self.gate_proj(x))
        x = x * gate
        x = self.mlp_down(F.silu(self.mlp_up(x)))
        x = self.dropout(x)
        return x, new_buf

    def _init_state(self, B: int, device: torch.device, dtype: torch.dtype) -> MosaicState:
        d = int(self.config.d_model)
        k = int(self.config.conv_kernel)

        conv_buf = torch.zeros((B, max(0, k - 1), d), device=device, dtype=dtype)
        s = torch.zeros((B, self.state_k, d), device=device, dtype=dtype)
        regs = torch.zeros((B, self.reg_slots, d), device=device, dtype=dtype) if self.reg_slots > 0 else None
        step = 0

        mem_k = torch.zeros(
            (B, self.memory.mem_hashes, self.memory.mem_buckets, self.memory.mem_assoc, self.memory.mem_key_dim),
            device=device, dtype=dtype
        )
        mem_v = torch.zeros(
            (B, self.memory.mem_hashes, self.memory.mem_buckets, self.memory.mem_assoc, self.memory.mem_dim),
            device=device, dtype=dtype
        )
        mem_last = torch.full(
            (B, self.memory.mem_hashes, self.memory.mem_buckets, self.memory.mem_assoc),
            -1, device=device, dtype=torch.long
        )

        return MosaicState(conv_buf=conv_buf, s=s, regs=regs, step=step, mem_k=mem_k, mem_v=mem_v, mem_last=mem_last)

    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        B, T, D = x.shape
        if D != int(self.config.d_model):
            raise ValueError(f"Expected d_model={int(self.config.d_model)}, got {D}")

        st = get_state(ctx, self._ctx_key)
        if st is not None:
             # Validate state shape match
             bad_regs = bool(isinstance(st.regs, Tensor) and st.regs.size(0) != B)
             if st.conv_buf.size(0) != B or st.s.size(0) != B or st.mem_k.size(0) != B or bad_regs:
                 st = None

        if st is None:
            st = self._init_state(B, x.device, x.dtype)

        u = _rms_norm(x)
        local, new_buf = self._local_mixer(u, ctx_state=st)
        if new_buf is not None:
            st.conv_buf = new_buf

        # Teacher forcing and dropout setup
        teacher = getattr(ctx, "mosaic_teacher", None) if ctx else None
        teacher_p = float(getattr(ctx, "mosaic_teacher_p", 1.0)) if ctx else 1.0
        teacher_p = max(0.0, min(1.0, teacher_p))

        self._apply_forced_read_dropout(local, ctx, B, T, x.device, x.dtype)

        # Precompute routing for the whole sequence
        collect_aux = bool(getattr(ctx, "mosaic_collect_aux", False))
        routing = self.memory.compute_routing(u, collect_aux=collect_aux)

        # Apply teacher overrides to routing if needed
        self._apply_routing_overrides(routing, teacher, teacher_p, B, T, x.device)

        # Teacher override for write gate (optional): passed to MosaicMemory.write_chunk.
        write_mask: Tensor | None = None
        if isinstance(teacher, dict) and "write_gate" in teacher:
            twg = teacher["write_gate"]
            if not isinstance(twg, Tensor):
                raise TypeError(f"Expected teacher['write_gate'] to be a Tensor, got {type(twg).__name__}")
            if twg.shape != (B, T):
                raise ValueError(f"teacher['write_gate'] must have shape (B,T)={(B,T)}, got {tuple(twg.shape)}")
            wg = twg.to(device=x.device, dtype=torch.float32)
            if teacher_p < 1.0:
                use = (torch.rand((B,), device=x.device) < float(teacher_p)).view(B, 1)
                wg = torch.where(use, wg, torch.full_like(wg, -1.0))
            write_mask = wg

        # Decide path
        stats_enabled = bool(ctx is not None and bool(getattr(ctx, "mosaic_stats_enabled", False)))
        has_clear = bool(isinstance(teacher, dict) and ("clear" in teacher))
        use_fast_train = (
            bool(self.training)
            and int(T) > 1
            and (not stats_enabled)
            and (not has_clear)
            and (self.reg_slots <= 0)
        )

        outputs: dict[str, Any] = {}
        opcode_probs: Tensor | None = None
        if self.opcodes_enabled and (bool(collect_aux) or bool(self.opcodes_control_enabled)):
            assert self.opcode_head is not None
            opcode_logits = self.opcode_head(u)
            if bool(collect_aux):
                outputs["opcode_logits"] = opcode_logits
            if bool(self.opcodes_control_enabled):
                temp = float(self.opcodes_control_temp)
                if temp <= 0.0:
                    raise ValueError(f"opcodes_control_temp must be > 0, got {temp}")
                opcode_probs = torch.softmax((opcode_logits.float() / float(temp)), dim=-1).to(dtype=u.dtype)
        if bool(collect_aux):
            # Keep learned decays within a healthy band to prevent early saturation.
            # (Proposed in meeting notes: keep decay rates in [0.001, 0.999].)
            dec = torch.sigmoid(self.state_decay_logit).to(dtype=torch.float32)
            lo = float(self.state_decay_reg_min)
            hi = float(self.state_decay_reg_max)
            if not (0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0 and lo <= hi):
                raise ValueError(f"Invalid state_decay_reg range: min={lo}, max={hi}")
            # Hinge penalty outside [lo, hi], averaged across timescales.
            pen = (F.relu(lo - dec) + F.relu(dec - hi)).mean()
            outputs["state_decay_reg_loss"] = pen
        if use_fast_train:
             delta = self._fast_train_path(u, local, st, routing, teacher, teacher_p, write_mask, opcode_probs, ctx, outputs)
        else:
             delta = self._sequential_path(u, local, st, routing, teacher, teacher_p, write_mask, opcode_probs, ctx, outputs)

        y = x + delta

        # Save state
        set_state(ctx, self._ctx_key, st)

        # Handle aux outputs
        if collect_aux:
            self._save_aux_outputs(ctx, outputs, st, routing)

        return y

    def _apply_forced_read_dropout(self, local: Tensor, ctx: Any, B: int, T: int, device: Any, dtype: Any):
        drop_p = float(getattr(self.config, "forced_read_dropout_p", 0.0))
        drop_mask: Tensor | None = None

        if ctx is not None:
            dm = getattr(ctx, "mosaic_drop_local", None)
            if isinstance(dm, Tensor):
                try:
                    if dm.dim() == 2 and dm.size(0) == B and dm.size(1) == T:
                        drop_mask = dm.to(device=device, dtype=dtype).view(B, T, 1)
                    elif dm.dim() == 1 and dm.size(0) == T:
                        drop_mask = dm.to(device=device, dtype=dtype).view(1, T, 1).expand(B, T, 1)
                except Exception:
                    pass

        if drop_mask is None and self.training and drop_p > 0.0:
            drop_mask = (torch.rand((B, T, 1), device=device) < drop_p).to(dtype=dtype)

        if drop_mask is not None:
            local.mul_(1.0 - drop_mask)

    def _apply_routing_overrides(self, routing: dict[str, Any], teacher: Any, teacher_p: float, B: int, T: int, device: Any):
        if teacher is None:
            return

        if "read_bucket" in teacher:
             tb = teacher["read_bucket"]
             if isinstance(tb, Tensor) and tb.dim() == 3: # (B, T, H)
                 use = (tb >= 0)
                 if teacher_p < 1.0:
                      mask = torch.rand((B,), device=device) < teacher_p
                      use = use & mask.view(B, 1, 1)
                 routing["idx_r"] = torch.where(use, tb.to(dtype=routing["idx_r"].dtype), routing["idx_r"])

        if "write_bucket" in teacher:
             tbw = teacher["write_bucket"]
             if isinstance(tbw, Tensor) and tbw.dim() == 3:
                 use = (tbw >= 0)
                 if teacher_p < 1.0:
                      mask = torch.rand((B,), device=device) < teacher_p
                      use = use & mask.view(B, 1, 1)
                 routing["idx_w"] = torch.where(use, tbw.to(dtype=routing["idx_w"].dtype), routing["idx_w"])

    def _fast_train_path(
        self, u: Tensor, local: Tensor, st: MosaicState, routing: dict[str, Any],
        teacher: Any, teacher_p: float, write_mask: Tensor | None, opcode_probs: Tensor | None, ctx: Any, outputs: dict[str, Any]
    ) -> Tensor:
        B, T, D = u.shape
        chunk_size = int(getattr(self.config, "mem_train_chunk_size", 128))
        chunk_size = max(1, chunk_size)

        decay = torch.sigmoid(self.state_decay_logit).to(dtype=u.dtype, device=u.device).view(1, self.state_k, 1)

        g_parts = []
        r_parts = []
        gate_logits_parts = []
        util_logits_parts = []

        s = st.s

        for t0 in range(0, int(T), chunk_size):
            t1 = min(int(T), t0 + chunk_size)
            u_c = u[:, t0:t1, :]
            C = u_c.size(1)

            # State bank
            inp = self.state_in(u_c).view(B, C, self.state_k, D).permute(0, 2, 1, 3)
            s_seq_c, s_last = leaky_integrator_scan(inp, s, decay)
            s = s_last.to(dtype=u.dtype)
            g_c = self.state_out(s_seq_c.permute(0, 2, 1, 3).reshape(B, C, self.state_k * D))
            g_parts.append(g_c)

            # Utility
            util_logit_c = self.memory.mem_utility_head(u_c).squeeze(-1)
            util_logits_parts.append(util_logit_c)

            # Slice routing
            routing_c = {k: v[:, t0:t1] if isinstance(v, Tensor) else v for k, v in routing.items()}

            # Memory Read
            r_c = self.memory.read(u_c, st, routing_c)
            r_parts.append(r_c)

            # Memory Write
            mask_c = write_mask[:, t0:t1] if isinstance(write_mask, Tensor) else None
            ws_c = (
                opcode_probs[:, t0:t1, int(MosaicOpcode.WRITE_MEM)]
                if isinstance(opcode_probs, Tensor)
                else None
            )
            gate_logit_c = self.memory.write_chunk(u_c, st, routing_c, t0, mask_c, write_scale=ws_c)
            gate_logits_parts.append(gate_logit_c)

        st.s = s.detach() # Persist state
        st.step += T

        # Concat parts
        g_seq = torch.cat(g_parts, dim=1)
        r_seq = torch.cat(r_parts, dim=1)

        # Fusion
        gate_long = torch.sigmoid(self.gate_long(u))
        gate_mem = torch.sigmoid(self.gate_mem(u))

        if isinstance(opcode_probs, Tensor):
            r_seq = r_seq * opcode_probs[:, :, int(MosaicOpcode.READ_MEM)].unsqueeze(-1)

        delta = local + gate_long * g_seq + gate_mem * r_seq

        # Store aux outputs
        outputs["gate_logits"] = torch.cat(gate_logits_parts, dim=1)
        outputs["util_logits"] = torch.cat(util_logits_parts, dim=1)

        return delta

    def _sequential_path(
        self, u: Tensor, local: Tensor, st: MosaicState, routing: dict[str, Any],
        teacher: Any, teacher_p: float, write_mask: Tensor | None, opcode_probs: Tensor | None, ctx: Any, outputs: dict[str, Any]
    ) -> Tensor:
        B, T, D = u.shape
        decay = torch.sigmoid(self.state_decay_logit).to(dtype=u.dtype, device=u.device).view(1, self.state_k, 1)

        g_parts = []
        r_parts = []
        z_parts = []

        # Aux collection
        gate_logits_parts = []
        util_logits_parts = []
        reg_gate_logits_parts = []
        reg_sel_logits_parts = []

        s = st.s
        regs = st.regs

        for t in range(int(T)):
            ut = u[:, t:t+1, :] # (B, 1, D)
            ws_t = (
                opcode_probs[:, t, int(MosaicOpcode.WRITE_MEM)]
                if isinstance(opcode_probs, Tensor)
                else None
            )

            # State bank update
            inp = self.state_in(ut).view(B, 1, self.state_k, D)
            s = decay * s + inp.squeeze(1)
            g_t = self.state_out(s.view(B, self.state_k * D))
            g_parts.append(g_t.unsqueeze(1))

            # Utility
            util_logit_t = self.memory.mem_utility_head(ut).squeeze(-1)
            util_logits_parts.append(util_logit_t)

            # Registers
            if self.reg_slots > 0:
                 zt, regs, reg_aux = self._update_registers(ut.squeeze(1), regs, write_scale=ws_t)
                 z_parts.append(zt.unsqueeze(1))
                 if reg_aux:
                     reg_gate_logits_parts.append(reg_aux[0])
                     reg_sel_logits_parts.append(reg_aux[1])

            # Routing for this step
            routing_t = {k: v[:, t:t+1] if isinstance(v, Tensor) else v for k, v in routing.items()}

            # Memory Read
            r_t = self.memory.read(ut, st, routing_t)
            r_parts.append(r_t)

            # Memory Write
            mask_t = write_mask[:, t:t+1] if isinstance(write_mask, Tensor) else None
            ws1 = ws_t.view(B, 1) if isinstance(ws_t, Tensor) else None
            gate_logit_t = self.memory.write_chunk(ut, st, routing_t, 0, mask_t, write_scale=ws1)
            gate_logits_parts.append(gate_logit_t)

            st.step += 1

        st.s = s.detach()
        if regs is not None:
             st.regs = regs.detach()

        g_seq = torch.cat(g_parts, dim=1)
        r_seq = torch.cat(r_parts, dim=1)

        gate_long = torch.sigmoid(self.gate_long(u))
        gate_mem = torch.sigmoid(self.gate_mem(u))

        if isinstance(opcode_probs, Tensor):
            r_seq = r_seq * opcode_probs[:, :, int(MosaicOpcode.READ_MEM)].unsqueeze(-1)

        delta = local + gate_long * g_seq + gate_mem * r_seq

        if self.reg_slots > 0 and len(z_parts) > 0:
             z_seq = torch.cat(z_parts, dim=1)
             assert self.gate_reg is not None
             gate_reg = torch.sigmoid(self.gate_reg(u))
             if isinstance(opcode_probs, Tensor):
                 z_seq = z_seq * opcode_probs[:, :, int(MosaicOpcode.WRITE_MEM)].unsqueeze(-1)
             delta = delta + gate_reg * z_seq

        # Aux outputs
        outputs["gate_logits"] = torch.cat(gate_logits_parts, dim=1)
        outputs["util_logits"] = torch.cat(util_logits_parts, dim=1)
        if len(reg_gate_logits_parts) > 0:
            outputs["reg_gate_logits"] = torch.stack(reg_gate_logits_parts, dim=1)
            outputs["reg_sel_logits"] = torch.stack(reg_sel_logits_parts, dim=1)

        return delta

    def _update_registers(
        self,
        u_t: Tensor,
        regs: Tensor | None,
        *,
        write_scale: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, tuple[Tensor, Tensor] | None]:
         if regs is None:
             return torch.zeros_like(u_t), None, None

         assert self.reg_write_gate is not None
         assert self.reg_sel is not None
         assert self.reg_value is not None

         # u_t: (B, D)
         # Read
         sim_r = torch.einsum("brd,bd->br", regs, u_t) * (1.0 / math.sqrt(float(u_t.size(-1))))
         w_r = torch.softmax(sim_r.float(), dim=-1).to(dtype=u_t.dtype)
         z_t = (w_r.unsqueeze(-1) * regs).sum(dim=1)

         # Write
         reg_gate_logit = self.reg_write_gate(u_t).squeeze(-1)
         sel_logits = self.reg_sel(u_t)

         aux = (reg_gate_logit, sel_logits)

         p_wr = torch.sigmoid(reg_gate_logit)
         if write_scale is not None:
             if not isinstance(write_scale, Tensor):
                 raise TypeError(f"write_scale must be a Tensor, got {type(write_scale).__name__}")
             if write_scale.ndim != 1 or write_scale.size(0) != u_t.size(0):
                 raise ValueError(
                     f"write_scale must have shape (B,)={(int(u_t.size(0)),)}, got {tuple(write_scale.shape)}"
                 )
             p_wr = p_wr * write_scale.to(dtype=p_wr.dtype, device=p_wr.device)
         thr_r = float(getattr(self.config, "reg_write_threshold", 0.5))
         eta_r = float(getattr(self.config, "reg_write_eta", 1.0))

         do_wr = p_wr > thr_r
         if do_wr.any():
              slot = sel_logits.argmax(dim=-1)
              val = self.reg_value(u_t)

              # Masked update
              b = torch.nonzero(do_wr, as_tuple=True)[0]
              s_idx = slot[b]
              v_wr = val[b]

              regs = regs.clone()
              if eta_r >= 0.999:
                   regs[b, s_idx, :] = v_wr
              else:
                   cur = regs[b, s_idx, :]
                   regs[b, s_idx, :] = (1.0 - eta_r) * cur + eta_r * v_wr

         return z_t, regs, aux

    def _save_aux_outputs(self, ctx: Any, outputs: dict, st: MosaicState, routing: dict):
         # Construct the aux dict
         aux = {}
         if "gate_logits" in outputs:
             aux["mosaic_write_gate_logits"] = outputs["gate_logits"]
         if "util_logits" in outputs:
             aux["mosaic_write_utility_logits"] = outputs["util_logits"]

         if "reg_gate_logits" in outputs:
             aux["mosaic_reg_write_gate_logits"] = outputs["reg_gate_logits"]
         if "reg_sel_logits" in outputs:
             aux["mosaic_reg_sel_logits"] = outputs["reg_sel_logits"]

         if st.regs is not None:
             aux["mosaic_regs_last"] = st.regs.detach()

         if "opcode_logits" in outputs:
             aux["mosaic_opcode_logits"] = outputs["opcode_logits"]
         if "state_decay_reg_loss" in outputs:
             aux["mosaic_state_decay_reg_loss"] = outputs["state_decay_reg_loss"]

         # Routing aux
         if "read_bit_logits" in routing:
             aux["mosaic_read_bit_logits"] = routing["read_bit_logits"]
         if "write_bit_logits" in routing:
             aux["mosaic_write_bit_logits"] = routing["write_bit_logits"]

         if "read_vq_logits" in routing:
             aux["mosaic_vq_read_logits"] = routing["read_vq_logits"]
         if "write_vq_logits" in routing:
             aux["mosaic_vq_write_logits"] = routing["write_vq_logits"]

         # Merge into ctx, accumulating scalar losses across stacked blocks.
         prev = getattr(ctx, "mosaic_aux_out", None)
         if isinstance(prev, dict):
             for k, v in aux.items():
                 if (
                     isinstance(k, str)
                     and isinstance(v, Tensor)
                     and int(v.numel()) == 1
                     and k.endswith("_loss")
                 ):
                     cur = prev.get(k)
                     if isinstance(cur, Tensor) and int(cur.numel()) == 1:
                         prev[k] = cur + v
                     else:
                         prev[k] = v
                 else:
                     prev[k] = v
             setattr(ctx, "mosaic_aux_out", prev)
         else:
             setattr(ctx, "mosaic_aux_out", aux)
