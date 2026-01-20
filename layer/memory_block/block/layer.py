"""Streaming memory block layer

This module implements a shape-preserving streaming block that relies on explicit
state (local buffers, multiscale integrators, and a hard-addressed memory table)
instead of attention/KV caches.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from config.layer import MemoryBlockLayerConfig
from instrumentation.training_metrics import get_training_metrics
from layer.memory_block.isa import MemoryOpcode
from layer.memory_block.memory import MemoryBlockMemory
from layer.memory_block.state import MemoryBlockState, MemoryBlockStateStore
from layer.memory_block.block.local_mixer import LocalMixer
from layer.memory_block.block.norm import RmsNorm
from layer.memory_block.block.path.fast_train import FastTrainPath
from layer.memory_block.block.path.sequential import SequentialPath
from layer.memory_block.block.state_bank import StateBank


def _mean_scalar(x: Tensor) -> Tensor:
    return x.detach().float().mean()


def _bucket_change_rate(idx: Tensor) -> Tensor:
    if idx.ndim != 3:
        return torch.zeros((), device=idx.device, dtype=torch.float32)
    if int(idx.size(1)) <= 1:
        return torch.zeros((), device=idx.device, dtype=torch.float32)
    d = (idx[:, 1:, :] != idx[:, :-1, :]).detach().float().mean()
    return d.to(dtype=torch.float32)


def _bucket_entropy_norm(idx: Tensor, buckets: int) -> Tensor:
    if idx.ndim != 3:
        return torch.zeros((), device=idx.device, dtype=torch.float32)
    v = idx.detach().reshape(-1).to(dtype=torch.long)
    v = v.clamp(0, int(buckets) - 1)
    c = torch.bincount(v, minlength=int(buckets)).float()
    p = c / c.sum().clamp_min(1.0)
    ent = -(p * (p.clamp_min(1e-12).log())).sum()
    denom = float(max(1.0, math.log(float(buckets))))
    return (ent / denom).to(dtype=torch.float32)


@dataclass(frozen=True, slots=True)
class OpcodeControl:
    """Opcode control surface.

    Uses straight-through selection (hard forward, soft backward) for gating.
    """

    vocab: int
    temp: float

    def control(self, logits: Tensor, *, dtype: torch.dtype) -> Tensor:
        """Compute opcode selections

        Straight-through selection makes the model commit to a discrete action
        in the forward pass while still getting a usable gradient signal during
        training.
        """
        if logits.ndim != 3:
            raise ValueError(f"opcode logits must have shape (B,T,V), got {tuple(logits.shape)}")
        if float(self.temp) <= 0.0:
            raise ValueError("OpcodeControl.temp must be > 0")
        soft = torch.softmax((logits.float() / float(self.temp)), dim=-1)
        idx = soft.argmax(dim=-1)
        hard = torch.nn.functional.one_hot(idx, num_classes=int(self.vocab)).to(dtype=soft.dtype)
        sel = (hard - soft).detach() + soft
        return sel.to(dtype=dtype)


class MemoryBlockLayer(nn.Module):
    """Streaming memory block layer

    This is a shape-preserving residual block for streaming models: it takes
    (B,T,D) in and returns (B,T,D), while maintaining per-layer state in the
    caller's context so decoding can be truly incremental.
    """

    def __init__(self, config: MemoryBlockLayerConfig) -> None:
        super().__init__()
        self.config = config
        self.ctx_key = f"memblock::{id(self)}"
        self.state_store = MemoryBlockStateStore()
        self.norm = RmsNorm(eps=1e-6)
        self.d_model = int(config.d_model)

        self.local_mixer = self.build_local_mixer()
        self.state_bank = self.build_state_bank()
        self.memory = MemoryBlockMemory(config, int(self.d_model))

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
        """Build the local mixer submodule

        The local mixer handles short-range pattern modeling with a fixed window,
        freeing the explicit memory subsystem to focus on longer-horizon storage.
        """
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
        """Build the multiscale state bank

        A bank of leaky integrators gives the block a set of learned time scales,
        which is a simple way to keep persistent intent without attention.
        """
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

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> MemoryBlockState:
        """Initialize streaming state

        The state contains the minimum buffers needed to make training and
        single-token decoding share the same semantics (conv history, integrator
        state, and explicit memory tables).
        """
        B = int(batch_size)
        D = int(self.d_model)
        k = int(getattr(self.config, "conv_kernel", 7))
        conv_buf = torch.zeros((B, max(0, k - 1), D), device=device, dtype=dtype)
        s = torch.zeros((B, int(self.state_bank.state_k), D), device=device, dtype=dtype)
        table_buckets = int(getattr(self.memory, "mem_table_buckets", self.memory.mem_buckets))
        shape_k = (B, self.memory.mem_hashes, table_buckets, self.memory.mem_assoc, self.memory.mem_key_dim)
        shape_v = (B, self.memory.mem_hashes, table_buckets, self.memory.mem_assoc, self.memory.mem_dim)
        shape_t = (B, self.memory.mem_hashes, table_buckets, self.memory.mem_assoc, self.memory.mem_vsa_dim)
        shape_l = (B, self.memory.mem_hashes, table_buckets, self.memory.mem_assoc)

        init_mode = str(getattr(self.config, "mem_init_mode", "empty")).lower().strip()
        init_scale = float(getattr(self.config, "mem_init_scale", 0.02))
        if init_scale <= 0.0:
            warnings.warn(
                f"Invalid mem_init_scale={init_scale}; falling back to 0.02",
                RuntimeWarning,
                stacklevel=2,
            )
            init_scale = 0.02

        allowed_modes = {"random_full", "random_empty", "zeros_full", "zeros_empty", "empty"}
        if init_mode not in allowed_modes:
            raise ValueError(
                f"Invalid mem_init_mode={init_mode!r}. Expected one of {sorted(allowed_modes)}."
            )

        if init_mode in {"random_full", "random_empty"}:
            mem_k = torch.randn(shape_k, device=device, dtype=dtype) * float(init_scale)
            mem_v = torch.randn(shape_v, device=device, dtype=dtype) * float(init_scale)
            mem_tag = torch.randn(shape_t, device=device, dtype=dtype) * float(init_scale)
        else:
            mem_k = torch.zeros(shape_k, device=device, dtype=dtype)
            mem_v = torch.zeros(shape_v, device=device, dtype=dtype)
            mem_tag = torch.zeros(shape_t, device=device, dtype=dtype)

        if init_mode in {"random_full", "zeros_full"}:
            mem_last = torch.zeros(shape_l, device=device, dtype=torch.long)
        else:
            mem_last = torch.full(shape_l, -1, device=device, dtype=torch.long)
        return MemoryBlockState(conv_buf=conv_buf, s=s, regs=None, step=0, mem_k=mem_k, mem_v=mem_v, mem_tag=mem_tag, mem_last=mem_last)

    def forward(self, x: Tensor, *, ctx: Any | None = None) -> Tensor:
        """Apply the MOSAIC block

        The block mixes local features, multiscale state, and explicit memory,
        then adds the result back to the input as a residual update.
        """
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B,T,D), got {tuple(x.shape)}")
        B, T, D = x.shape
        if int(D) != int(self.d_model):
            raise ValueError(f"Expected d_model={int(self.d_model)}, got {int(D)}")
        st = self.state_store.get(ctx, key=self.ctx_key)
        if st is None or int(st.s.size(0)) != int(B):
            st = self.init_state(int(B), x.device, x.dtype)

        u = self.norm.apply(x)
        local, new_buf = self.local_mixer.forward(u, state=st)
        if new_buf is not None:
            st.conv_buf = new_buf

        collect_aux = bool(getattr(ctx, "memblock_collect_aux", False)) if ctx is not None else False

        # Determine current step
        step = int(getattr(ctx, "step", 0) or 0)
        if ctx is None:
            # If ctx is missing (performance mode), try to get step from global metrics
            metrics = get_training_metrics()
            # Note: metrics.step is usually valid, defaulting to 0 if not set.
            # If metrics.step is 0, we might truly be at step 0 (warmup), which is fine.
            step = metrics.step

        # --- Visibility Refinement ---
        # Force collect_aux if visualization is enabled and we are in warmup or on an interval.
        # This ensures the tuner has data to display.
        if bool(getattr(self.config, "mem_autotune_viz", False)):
            viz_interval = int(getattr(self.config, "mem_autotune_viz_interval", 100))
            warmup_threshold = int(getattr(self.config, "mem_autotune_viz_warmup", 5))
            if step <= warmup_threshold or (step > 0 and step % viz_interval == 0):
                collect_aux = True

        teacher = getattr(ctx, "memblock_teacher", None) if ctx is not None else None
        teacher_p = float(getattr(ctx, "memblock_teacher_p", 1.0)) if ctx is not None else 0.0
        if (
            ctx is not None
            and isinstance(teacher, dict)
            and bool(getattr(self.memory, "rmf_enabled", False))
            and getattr(self.memory, "rmf", None) is not None
            and "read_bucket" in teacher
        ):
            routing = self.memory.compute_routing_with_teacher(u, st, teacher, collect_aux=collect_aux)
        else:
            routing = self.memory.compute_routing(u, collect_aux=collect_aux)
        routing["collect_aux"] = bool(collect_aux)
        routing["global_step"] = step

        # Pass loss from viz_ctx to routing (for tuner)
        if ctx is not None and hasattr(ctx, '_last_loss'):
            routing["_last_loss"] = ctx._last_loss

        # Pass training metrics from context to routing (for tuner)
        if ctx is not None:
            for name in ("train_accuracy", "train_loss", "train_loss_variance"):
                val = getattr(ctx, name, None)
                if val is not None:
                    routing[name] = val

        if ctx is not None and isinstance(teacher, dict):
            self.memory.apply_teacher_overrides(routing, teacher, p=float(teacher_p))
        write_mask = self.resolve_write_mask(ctx, B=B, T=T, device=x.device)
        # MOSAIC write warmup: force no-writes for first N training steps.
        if ctx is not None:
            warm = int(getattr(ctx, "memblock_write_warmup_steps", 0) or 0)
            step = int(getattr(ctx, "step", 0) or 0)
            if warm > 0 and step > 0 and step <= warm:
                write_mask = torch.zeros((int(B), int(T)), device=x.device, dtype=torch.float32)
        opcode_ctrl = self.compute_opcode_control(u, collect_aux=collect_aux)

        # Fast path is a training-only optimization, but it does not currently
        # model per-step RMF updates (which are stateful and order-dependent).
        # When RMF is enabled, prefer the exact sequential path so RMF state
        # (st.rmf_field) is updated and observable after forward.
        use_fast = (
            bool(self.training)
            and int(T) > 1
            and not bool(getattr(ctx, "memblock_stats_enabled", False))
            and not (bool(getattr(self.memory, "rmf_enabled", False)) and getattr(self.memory, "rmf", None) is not None)
        )
        if use_fast:
            delta, outputs = self.fast_path.run(u=u, local=local, st=st, routing=routing, write_mask=write_mask, opcode_ctrl=opcode_ctrl)
        else:
            delta, outputs = self.seq_path.run(u=u, local=local, st=st, routing=routing, write_mask=write_mask, opcode_ctrl=opcode_ctrl)

        y = x + delta
        self.state_store.set(ctx, key=self.ctx_key, state=st)
        if collect_aux and ctx is not None:
            self.save_aux(ctx, outputs=outputs, routing=routing, opcode_logits=self.get_opcode_logits(u))
        return y

    def resolve_write_mask(self, ctx: Any | None, *, B: int, T: int, device: torch.device) -> Tensor | None:
        """Resolve an optional teacher-provided write mask

        A write mask is a training-time control signal that lets you supervise
        or restrict memory writes without changing the rest of the block logic.
        """
        teacher = getattr(ctx, "memblock_teacher", None) if ctx is not None else None
        if not (isinstance(teacher, dict) and "write_gate" in teacher):
            return None
        wg = teacher["write_gate"]
        if not isinstance(wg, Tensor) or tuple(wg.shape) != (int(B), int(T)):
            raise ValueError(f"teacher['write_gate'] must have shape (B,T)={(B,T)}, got {getattr(wg, 'shape', None)}")
        return wg.to(device=device, dtype=torch.float32)

    def compute_opcode_control(self, u: Tensor, *, collect_aux: bool) -> Tensor | None:
        """Compute opcode control tensor

        Opcodes act like a tiny “action vocabulary” the block can use to gate
        reads/writes; this makes control decisions inspectable and, if desired,
        supervisable.
        """
        if not bool(self.opcodes_enabled) or self.opcode_head is None:
            return None
        if not bool(self.opcodes_control_enabled):
            return None
        ctrl = OpcodeControl(vocab=int(self.opcode_vocab), temp=float(self.opcodes_control_temp))
        return ctrl.control(self.opcode_head(u), dtype=u.dtype)

    def get_opcode_logits(self, u: Tensor) -> Tensor | None:
        """Get opcode logits

        Exposing raw logits is useful for instrumentation and supervised
        training targets without threading extra outputs through the forward.
        """
        if not bool(self.opcodes_enabled) or self.opcode_head is None:
            return None
        return self.opcode_head(u)

    class _AuxCtx(Protocol):
        memblock_aux_out: dict[str, Tensor] | None

    def save_aux(
        self,
        ctx: _AuxCtx,
        *,
        outputs: dict[str, Tensor],
        routing: dict[str, Any],
        opcode_logits: Tensor | None,
    ) -> None:
        """Save auxiliary outputs into the context

        The block produces rich internal signals (gates, routing logits, stats);
        storing them on the context keeps the layer API clean while still
        enabling debugging and research instrumentation.
        """
        aux = getattr(ctx, "memblock_aux_out", None)
        if aux is None:
            aux = {}
        if not isinstance(aux, dict):
            raise TypeError("ctx.memblock_aux_out must be a dict when collecting aux")
        aux["memblock_write_gate_logits"] = outputs["gate_logits"]
        aux["memblock_write_utility_logits"] = outputs["util_logits"]
        if isinstance(opcode_logits, Tensor):
            aux["memblock_opcode_logits"] = opcode_logits
        if "read_bit_logits" in routing:
            aux["memblock_read_bit_logits"] = routing["read_bit_logits"]
        if "write_bit_logits" in routing:
            aux["memblock_write_bit_logits"] = routing["write_bit_logits"]
        if "read_vq_logits" in routing:
            aux["memblock_vq_read_logits"] = routing["read_vq_logits"]
        if "write_vq_logits" in routing:
            aux["memblock_vq_write_logits"] = routing["write_vq_logits"]
        ctx.memblock_aux_out = aux

        # RMF observability: cheap scalar stats for logging.
        if not bool(getattr(ctx, "memblock_stats_enabled", False)):
            return
        mem_stats = getattr(ctx, "memblock_mem_stats", None)
        if not isinstance(mem_stats, dict):
            return

        if "rmf_delta_rms" in routing and isinstance(routing["rmf_delta_rms"], Tensor):
            mem_stats[f"{self.ctx_key}/rmf_delta_rms"] = _mean_scalar(routing["rmf_delta_rms"])
        if "rmf_field_rms" in routing and isinstance(routing["rmf_field_rms"], Tensor):
            mem_stats[f"{self.ctx_key}/rmf_field_rms"] = _mean_scalar(routing["rmf_field_rms"])

        # Teacher addressing diagnostics: agreement on non-teacher-forced steps.
        if "read_teacher_agree" in routing and isinstance(routing["read_teacher_agree"], Tensor):
            mem_stats[f"{self.ctx_key}/read_teacher_agree"] = _mean_scalar(routing["read_teacher_agree"])
        if "read_teacher_agree_free" in routing and isinstance(routing["read_teacher_agree_free"], Tensor):
            mem_stats[f"{self.ctx_key}/read_teacher_agree_free"] = _mean_scalar(routing["read_teacher_agree_free"])
        if "write_teacher_agree" in routing and isinstance(routing["write_teacher_agree"], Tensor):
            mem_stats[f"{self.ctx_key}/write_teacher_agree"] = _mean_scalar(routing["write_teacher_agree"])
        if "write_teacher_agree_free" in routing and isinstance(routing["write_teacher_agree_free"], Tensor):
            mem_stats[f"{self.ctx_key}/write_teacher_agree_free"] = _mean_scalar(routing["write_teacher_agree_free"])
        if "read_teacher_agree_label_count" in routing and isinstance(routing["read_teacher_agree_label_count"], Tensor):
            mem_stats[f"{self.ctx_key}/read_teacher_label_count"] = routing["read_teacher_agree_label_count"].detach().float()
        if "read_teacher_agree_probe_count" in routing and isinstance(routing["read_teacher_agree_probe_count"], Tensor):
            mem_stats[f"{self.ctx_key}/read_teacher_probe_count"] = routing["read_teacher_agree_probe_count"].detach().float()
        if "write_teacher_agree_label_count" in routing and isinstance(routing["write_teacher_agree_label_count"], Tensor):
            mem_stats[f"{self.ctx_key}/write_teacher_label_count"] = routing["write_teacher_agree_label_count"].detach().float()
        if "write_teacher_agree_probe_count" in routing and isinstance(routing["write_teacher_agree_probe_count"], Tensor):
            mem_stats[f"{self.ctx_key}/write_teacher_probe_count"] = routing["write_teacher_agree_probe_count"].detach().float()
        if "teacher_used_frac" in routing and isinstance(routing["teacher_used_frac"], Tensor):
            mem_stats[f"{self.ctx_key}/teacher_used_frac"] = _mean_scalar(routing["teacher_used_frac"])

        # Routing dynamics diagnostics.
        idx_r_src = routing.get("idx_r_pre", routing.get("idx_r", None))
        if isinstance(idx_r_src, Tensor):
            mem_stats[f"{self.ctx_key}/read_bucket_change_rate"] = _bucket_change_rate(idx_r_src)
        idx_w_src = routing.get("idx_w_pre", routing.get("idx_w", None))
        if isinstance(idx_w_src, Tensor):
            mem_stats[f"{self.ctx_key}/write_bucket_change_rate"] = _bucket_change_rate(idx_w_src)
            mem_stats[f"{self.ctx_key}/write_bucket_entropy_norm"] = _bucket_entropy_norm(
                idx_w_src, int(getattr(self.memory, "mem_buckets", 1024))
            )

        # VQ router accuracy (early signal): per-group code accuracy is far more informative than
        # full bucket match in high-cardinality tables.
        if "vq_read_group_acc" in routing and isinstance(routing["vq_read_group_acc"], Tensor):
            mem_stats[f"{self.ctx_key}/vq_read_group_acc"] = routing["vq_read_group_acc"].detach().float()
        if "vq_write_group_acc" in routing and isinstance(routing["vq_write_group_acc"], Tensor):
            mem_stats[f"{self.ctx_key}/vq_write_group_acc"] = routing["vq_write_group_acc"].detach().float()

        # Write gate behavior.
        gate_logits = outputs.get("gate_logits", None)
        if isinstance(gate_logits, Tensor):
            p = torch.sigmoid(gate_logits.detach().float())
            mem_stats[f"{self.ctx_key}/write_gate_p_mean"] = p.mean()
            thr = float(getattr(self.memory, "mem_write_threshold", 0.5))
            mem_stats[f"{self.ctx_key}/write_gate_fire_frac"] = (p > float(thr)).float().mean()

