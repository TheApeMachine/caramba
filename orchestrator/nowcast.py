"""High-performance weight nowcasting (platform-grade).

Design goals:
- Nodes are *parameter tensors* (or blocks of tensors), not scalar weights.
- No per-step full snapshots or huge training buffers.
- Online training with a sketch-based loss (fast, low-memory).
- Static cached graph built from parameter/module structure.
- Minimal device sync: avoid per-parameter `.item()` in hot paths.
- Optional optimizer-state policy on nowcast to reduce instability.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Callable, Literal

import torch
from torch import Tensor, nn

from caramba.carmath.sketch import sketch_dot5, stride_sketch_indices
from caramba.optimizer.runtime import triton_supported

log = logging.getLogger(__name__)

# ---------------------------
# Configuration
# ---------------------------

OptimizerStatePolicy = Literal["keep", "reset_tensors", "reset_all"]
BlockNodeMode = Literal["tensor", "out_channels"]
EmaSyncPolicy = Literal["none", "copy_model"]


@dataclass(slots=True)
class NowcastConfig:
    # Forecasting
    horizon: int = 50
    nowcast_interval: int = 100
    min_steps_before_start: int = 200

    # Online predictor training
    train_predictor_every: int = 10
    predictor_lr: float = 1e-4
    predictor_weight_decay: float = 1e-5

    # Quality gating (based on 1-step relative prediction error)
    max_forecast_error: float = 0.10
    error_window: int = 20
    min_active_params_for_error: int = 8  # require enough active nodes

    # History of scalar signals (cheap)
    history_size: int = 20  # for mean/std features

    # Sketching: compute dot-products/norms on a fixed subsample per node.
    use_sketch: bool = True
    sketch_size: int = 2048
    sketch_min_numel: int = 4096  # below this, use full node tensor
    sketch_stride_hash: bool = True

    # Graph construction (static)
    use_module_edges: bool = True
    use_parent_edges: bool = True
    max_edges_per_group: int = 32
    use_block_chain_edges: bool = True
    block_chain_edge_weight: float = 0.25

    # Block nodes (optional, off the hot path by default)
    block_node_mode: BlockNodeMode = "tensor"
    block_size: int = 256  # out_channels per node when block_node_mode="out_channels"
    block_min_dim0: int = 2048  # only block if dim0 is large enough
    max_block_nodes_per_param: int = 2048  # safety cap; fallback to tensor mode if exceeded

    # Predictor architecture
    node_embed_dim: int = 64
    hidden_dim: int = 128
    num_gnn_layers: int = 3
    num_head_layers: int = 2

    # Output constraint for scale 'a': a = 1 + scale_range * tanh(raw)
    scale_range: float = 1.5

    # Precision/compile (predictor only)
    use_amp: bool = True
    prefer_bf16: bool = True
    compile_predictor: bool = False

    # Main-optimizer state handling on nowcast
    optimizer_state_policy: OptimizerStatePolicy = "reset_tensors"
    advance_optimizer_steps_on_nowcast: bool = True  # Adam/AdamW bias correction step advance

    # EMA sync (optional)
    ema_sync_policy: EmaSyncPolicy = "none"


def _bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


# ---------------------------
# Graph + Predictor
# ---------------------------


class InteractionGNN(nn.Module):
    """Lightweight gated message passing."""

    def __init__(self, dim: int, hidden: int, layers: int) -> None:
        super().__init__()
        self.layers = int(layers)
        self.edge_mlps = nn.ModuleList()
        self.node_mlps = nn.ModuleList()
        self.lns = nn.ModuleList()

        for _ in range(self.layers):
            self.edge_mlps.append(nn.Sequential(
                nn.Linear(dim * 2 + 1, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
                nn.Sigmoid(),
            ))
            self.node_mlps.append(nn.Sequential(
                nn.Linear(dim * 2, hidden),
                nn.GELU(),
                nn.Linear(hidden, dim),
            ))
            self.lns.append(nn.LayerNorm(dim))

    def forward(self, h: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # h: (N, D), edge_index: (E, 2), edge_attr: (E, 1)
        src = edge_index[:, 0]
        dst = edge_index[:, 1]

        for i in range(self.layers):
            h_src = h[src]
            h_dst = h[dst]
            gate = self.edge_mlps[i](torch.cat([h_src, h_dst, edge_attr], dim=-1))  # (E,1)
            msg = h_src * gate  # (E,D)

            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, msg)

            den = torch.zeros((h.shape[0], 1), device=h.device, dtype=h.dtype)
            den.index_add_(0, dst, gate)
            agg = agg / den.clamp_min(1e-6)

            upd = self.node_mlps[i](torch.cat([h, agg], dim=-1))
            h = self.lns[i](h + upd)

        return h


class NowcastNet(nn.Module):
    """Maps node features to a per-node update scale a."""

    def __init__(self, in_dim: int, cfg: NowcastConfig) -> None:
        super().__init__()
        self.scale_range = float(cfg.scale_range)

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, cfg.node_embed_dim),
            nn.LayerNorm(cfg.node_embed_dim),
            nn.GELU(),
            nn.Linear(cfg.node_embed_dim, cfg.node_embed_dim),
        )

        self.gnn = InteractionGNN(cfg.node_embed_dim, cfg.hidden_dim, cfg.num_gnn_layers)

        head: list[nn.Module] = []
        d = cfg.node_embed_dim
        for _ in range(max(1, int(cfg.num_head_layers)) - 1):
            head += [nn.Linear(d, cfg.hidden_dim), nn.GELU()]
            d = cfg.hidden_dim
        head += [nn.Linear(d, 1)]
        self.head = nn.Sequential(*head)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        h = self.encoder(x)
        h = self.gnn(h, edge_index, edge_attr)
        raw = self.head(h).squeeze(-1)  # (N,)
        return 1.0 + self.scale_range * torch.tanh(raw)  # (N,)


class WeightGraphBuilder:
    """Static graph between parameter(-block) nodes, derived from module hierarchy."""

    def __init__(self, cfg: NowcastConfig) -> None:
        self.cfg = cfg
        self._cached: tuple[Tensor, Tensor] | None = None
        self._cached_sig: tuple[str, ...] | None = None

    @staticmethod
    def _module_key(name: str) -> str:
        # name may include a "#b.." suffix; module parsing ignores that naturally.
        parts = name.split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else name

    @staticmethod
    def _parent_key(module_key: str) -> str:
        parts = module_key.split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else module_key

    def build(self, node_names: list[str], device: torch.device) -> tuple[Tensor, Tensor]:
        sig = tuple(node_names)
        if self._cached is not None and self._cached_sig == sig:
            return self._cached

        n = len(node_names)
        mod_keys = [self._module_key(nm) for nm in node_names]

        mod2nodes: dict[str, list[int]] = {}
        for i, k in enumerate(mod_keys):
            mod2nodes.setdefault(k, []).append(i)

        edges: list[tuple[int, int, float]] = []

        def add_edge(u: int, v: int, w: float) -> None:
            if u != v:
                edges.append((u, v, w))

        if self.cfg.use_module_edges:
            for _, nodes in mod2nodes.items():
                if len(nodes) <= 1:
                    continue
                hub = nodes[0]
                others = nodes[1 : 1 + int(self.cfg.max_edges_per_group)]
                for j in others:
                    add_edge(hub, j, 1.0)
                    add_edge(j, hub, 1.0)

        if self.cfg.use_parent_edges:
            parent2mods: dict[str, list[str]] = {}
            for mk in mod2nodes.keys():
                parent2mods.setdefault(self._parent_key(mk), []).append(mk)

            for _, mods in parent2mods.items():
                if len(mods) <= 1:
                    continue
                reps = [mod2nodes[mk][0] for mk in mods]
                hub = reps[0]
                for r in reps[1 : 1 + int(self.cfg.max_edges_per_group)]:
                    add_edge(hub, r, 0.5)
                    add_edge(r, hub, 0.5)

        if self.cfg.use_block_chain_edges:
            # Add locality edges between consecutive blocks for the same parameter.
            # This helps conv/embedding-style tensors where dim0 has meaningful adjacency.
            by_base: dict[str, list[tuple[int, int]]] = {}
            for i, nm in enumerate(node_names):
                if "#b" not in nm:
                    continue
                base, suffix = nm.split("#b", 1)
                try:
                    lo = int(suffix.split(":")[0])
                except (ValueError, IndexError):
                    log.warning("WeightNowcaster: skipping malformed block suffix %r in node %r", suffix, nm)
                    continue
                by_base.setdefault(base, []).append((lo, i))

            for _, blocks in by_base.items():
                blocks.sort(key=lambda t: t[0])
                for (_lo0, i0), (_lo1, i1) in pairwise(blocks):
                    add_edge(i0, i1, float(self.cfg.block_chain_edge_weight))
                    add_edge(i1, i0, float(self.cfg.block_chain_edge_weight))

        if not edges:
            for i in range(n - 1):
                edges.append((i, i + 1, 1.0))
                edges.append((i + 1, i, 1.0))

        edge_index = torch.tensor([(u, v) for (u, v, _) in edges], device=device, dtype=torch.long)
        edge_attr = torch.tensor([[w] for (_, _, w) in edges], device=device, dtype=torch.float32)

        self._cached = (edge_index, edge_attr)
        self._cached_sig = sig
        return edge_index, edge_attr


# ---------------------------
# Main Nowcaster
# ---------------------------


@dataclass(slots=True)
class _TrackedParam:
    name: str
    param: nn.Parameter
    w_prev: Tensor
    delta_prev: Tensor


@dataclass(slots=True)
class _NodeState:
    name: str
    tracked: _TrackedParam
    dim0_slice: slice | None
    sketch_idx: Tensor | None
    sketch_len: int
    log_numel: float

    def view(self, t: Tensor) -> Tensor:
        return t if self.dim0_slice is None else t[self.dim0_slice]


class WeightNowcaster:
    """High-performance weight nowcasting module.

    Predicts a per-node scale a for next-step update:
        Δw_t ≈ a * Δw_{t-1}

    Integration contract:
    - Call record(step) AFTER the real optimizer update and BEFORE gradients are cleared.
    - Call should_nowcast(step) then nowcast(step) occasionally.
    """

    # History signals stored per step:
    #   0: delta_rms (||Δw_t|| / sqrt(k))
    #   1: grad_rms  (||g_t|| / sqrt(k))
    #   2: cos_dp    cos(Δw_{t-1}, Δw_t)
    _HIST_DIM = 3

    def __init__(
        self,
        model: nn.Module,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        config: NowcastConfig | None = None,
        param_filter: Callable[[str, nn.Parameter], bool] | None = None,
        ema_model: nn.Module | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.ema_model = ema_model
        self.cfg = config or NowcastConfig()

        try:
            self.device = next(model.parameters()).device
        except StopIteration as e:
            raise ValueError("WeightNowcaster requires a model with parameters.") from e

        # Predictor dtype
        self._pred_dtype: torch.dtype | None = None
        if self.device.type == "cuda" and self.cfg.use_amp:
            if self.cfg.prefer_bf16 and _bf16_supported():
                self._pred_dtype = torch.bfloat16
            else:
                self._pred_dtype = torch.float16

        # Build tracked params + nodes
        self._tracked_params: list[_TrackedParam] = []
        self._nodes: list[_NodeState] = []
        self._build_nodes(param_filter)

        if not self._nodes:
            raise ValueError("WeightNowcaster found no trainable parameters to track.")

        self._n = len(self._nodes)

        # Cached graph for nodes
        self._graph = WeightGraphBuilder(self.cfg)
        self._edge_index, self._edge_attr = self._graph.build(
            [ns.name for ns in self._nodes],
            device=self.device,
        )

        # History buffer: (H, N, HIST_DIM)
        H = max(1, int(self.cfg.history_size))
        self._hist = torch.zeros((H, self._n, self._HIST_DIM), device=self.device, dtype=torch.float32)
        self._hist_i = 0
        self._hist_count = 0

        # Predictor
        in_dim = self._feature_dim()
        self._net = NowcastNet(in_dim, self.cfg).to(self.device)
        if self._pred_dtype is not None:
            self._net = self._net.to(dtype=self._pred_dtype)
        if self.cfg.compile_predictor and hasattr(torch, "compile"):
            self._net = torch.compile(self._net)  # type: ignore[attr-defined]

        self._opt = torch.optim.AdamW(
            self._net.parameters(),
            lr=float(self.cfg.predictor_lr),
            weight_decay=float(self.cfg.predictor_weight_decay),
        )

        # State
        self._enabled = True
        self._last_nowcast_step = -10**18
        self._step = -1

        # Error tracking (CPU deque; updated only on train intervals to avoid sync)
        self._errors: deque[float] = deque(maxlen=int(self.cfg.error_window))

        # Pre-allocated scalar buffers (float32, on device)
        self._uu = torch.zeros((self._n,), device=self.device, dtype=torch.float32)
        self._tt = torch.zeros((self._n,), device=self.device, dtype=torch.float32)
        self._ut = torch.zeros((self._n,), device=self.device, dtype=torch.float32)
        self._vv = torch.zeros((self._n,), device=self.device, dtype=torch.float32)
        self._uv = torch.zeros((self._n,), device=self.device, dtype=torch.float32)

        self._sketch_len = torch.tensor([ns.sketch_len for ns in self._nodes], device=self.device, dtype=torch.float32)
        self._log_numel = torch.tensor([ns.log_numel for ns in self._nodes], device=self.device, dtype=torch.float32)

    # ---------------------------
    # Public API
    # ---------------------------

    @property
    def enabled(self) -> bool:
        return bool(self._enabled)

    def enable(self) -> None:
        self._enabled = True
        self._errors.clear()

    def disable(self) -> None:
        self._enabled = False

    def record(self, step: int) -> None:
        """Record a real training update and (optionally) train the predictor."""
        self._step = int(step)

        if step <= 0:
            self._compute_step_stats()
            self._commit_history_and_state()
            return

        self._compute_step_stats()
        x = self._build_node_features()
        a = self._predict_scale(x)

        # Train / update error tracking only every N steps to avoid CPU sync.
        if int(self.cfg.train_predictor_every) > 0 and (step % int(self.cfg.train_predictor_every) == 0):
            self._update_error_and_maybe_train(a)

        self._commit_history_and_state()

    def should_nowcast(self, step: int) -> bool:
        step = int(step)
        if not self._enabled:
            return False
        if step < int(self.cfg.min_steps_before_start):
            return False
        if (step - self._last_nowcast_step) < int(self.cfg.nowcast_interval):
            return False
        if self._hist_count < 2:
            return False
        return True

    def nowcast(self, step: int) -> int:
        """Apply a nowcast jump to the model weights; returns steps skipped."""
        step = int(step)
        if not self.should_nowcast(step):
            return 0

        horizon = int(self.cfg.horizon)
        if horizon <= 0:
            return 0

        x = self._build_node_features()
        a = self._predict_scale(x)  # (N,) float32 on device

        with torch.no_grad():
            # Optimizer state policy (before modifying weights)
            if self.optimizer is not None:
                self._apply_optimizer_state_policy(self.optimizer, self.cfg.optimizer_state_policy)
                if self.cfg.advance_optimizer_steps_on_nowcast:
                    self._advance_optimizer_steps_if_supported(self.optimizer, horizon)

            # Apply per-node update and keep w_prev aligned
            for i, ns in enumerate(self._nodes):
                p = ns.tracked.param
                d = ns.view(ns.tracked.delta_prev)
                w = ns.view(p.data)

                # delta_prev <- a * delta_prev (keeps internal velocity consistent)
                d.mul_(a[i].to(dtype=d.dtype))

                # w <- w + horizon * delta_prev  (avoid alpha on predicted a)
                w.add_(d, alpha=float(horizon))

                # Keep w_prev aligned for next real record()
                ns.view(ns.tracked.w_prev).copy_(w)

            # Optional EMA sync
            if self.ema_model is not None and self.cfg.ema_sync_policy == "copy_model":
                self._sync_ema_model(self.ema_model)

        self._last_nowcast_step = step
        return horizon

    def get_stats(self) -> dict[str, Any]:
        errs = list(self._errors)
        return {
            "enabled": bool(self._enabled),
            "step": int(self._step),
            "last_nowcast_step": int(self._last_nowcast_step),
            "history_count": int(self._hist_count),
            "avg_error": (sum(errs) / len(errs)) if errs else 0.0,
            "max_error": (max(errs) if errs else 0.0),
            "num_nodes_tracked": int(self._n),
            "num_params_tracked": int(len(self._tracked_params)),
            "block_node_mode": str(self.cfg.block_node_mode),
            "predictor_dtype": str(self._pred_dtype) if self._pred_dtype is not None else "fp32",
            "triton_available": bool(triton_supported()),
        }

    # ---------------------------
    # Internals
    # ---------------------------

    def _build_nodes(self, param_filter: Callable[[str, nn.Parameter], bool] | None) -> None:
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if param_filter is not None and not bool(param_filter(name, p)):
                continue
            if p.numel() == 0:
                continue

            tracked = _TrackedParam(
                name=name,
                param=p,
                w_prev=p.data.detach().clone(),
                delta_prev=torch.zeros_like(p.data),
            )
            self._tracked_params.append(tracked)

            # Decide node decomposition
            if self.cfg.block_node_mode == "out_channels" and p.dim() >= 2 and p.shape[0] >= int(self.cfg.block_min_dim0):
                bsz = max(1, int(self.cfg.block_size))
                dim0 = int(p.shape[0])
                nblocks = (dim0 + bsz - 1) // bsz
                if nblocks > int(self.cfg.max_block_nodes_per_param):
                    # Safety fallback
                    self._nodes.append(self._make_node(tracked, name, None))
                    continue
                for bi in range(nblocks):
                    lo = bi * bsz
                    hi = min(dim0, (bi + 1) * bsz)
                    node_name = f"{name}#b{lo}:{hi}"
                    self._nodes.append(self._make_node(tracked, node_name, slice(lo, hi)))
            else:
                self._nodes.append(self._make_node(tracked, name, None))

    def _make_node(self, tracked: _TrackedParam, node_name: str, dim0_slice: slice | None) -> _NodeState:
        # Determine node tensor length (for sketch sizing) without allocating.
        if dim0_slice is None:
            numel = int(tracked.param.numel())
        else:
            # slice along dim0 => product of remaining dims
            per = int(tracked.param.numel() // max(1, int(tracked.param.shape[0])))
            numel = int((dim0_slice.stop - dim0_slice.start) * per)

        if (not self.cfg.use_sketch) or (numel < int(self.cfg.sketch_min_numel)) or (int(self.cfg.sketch_size) >= numel):
            idx = None
            k = numel
        else:
            k = int(self.cfg.sketch_size)
            idx = stride_sketch_indices(
                numel,
                k,
                seed=node_name,
                device=self.device,
                hashed_start=bool(self.cfg.sketch_stride_hash),
            )

        log_numel = float(math.log1p(max(1, numel)))
        return _NodeState(
            name=node_name,
            tracked=tracked,
            dim0_slice=dim0_slice,
            sketch_idx=idx,
            sketch_len=int(k),
            log_numel=log_numel,
        )

    def _feature_dim(self) -> int:
        # [log_numel,
        #  prev_delta_rms, grad_rms, cos_u_grad,
        #  hist_mean(delta_rms), hist_std(delta_rms),
        #  hist_mean(grad_rms),  hist_std(grad_rms),
        #  hist_mean(cos_dp),    hist_std(cos_dp)]
        return 10

    def _hist_view(self) -> Tensor:
        if self._hist_count < self._hist.shape[0]:
            return self._hist[: self._hist_count]
        return self._hist

    def _history_stats(self) -> tuple[Tensor, Tensor]:
        hv = self._hist_view()
        if hv.numel() == 0:
            mean = torch.zeros((self._n, self._HIST_DIM), device=self.device, dtype=torch.float32)
            std = torch.zeros((self._n, self._HIST_DIM), device=self.device, dtype=torch.float32)
            return mean, std
        mean = hv.mean(dim=0)
        std = hv.std(dim=0, unbiased=False)
        return mean, std

    def _build_node_features(self) -> Tensor:
        eps = 1e-8
        mean, std = self._history_stats()

        prev_delta_rms = torch.sqrt(self._uu / self._sketch_len.clamp_min(1.0))
        grad_rms = torch.sqrt(self._vv / self._sketch_len.clamp_min(1.0))
        cos_u_grad = self._uv / torch.sqrt((self._uu * self._vv).clamp_min(eps))

        x = torch.cat([
            self._log_numel[:, None],
            prev_delta_rms[:, None],
            grad_rms[:, None],
            cos_u_grad[:, None],
            mean[:, 0:1], std[:, 0:1],
            mean[:, 1:2], std[:, 1:2],
            mean[:, 2:3], std[:, 2:3],
        ], dim=1)
        return x.to(dtype=torch.float32)

    def _predict_scale(self, x: Tensor) -> Tensor:
        if self._pred_dtype is not None and self.cfg.use_amp and self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=self._pred_dtype):
                a = self._net(x.to(dtype=self._pred_dtype), self._edge_index, self._edge_attr)
            return a.to(dtype=torch.float32)
        return self._net(x, self._edge_index, self._edge_attr).to(dtype=torch.float32)

    def _update_error_and_maybe_train(self, a: Tensor) -> None:
        eps = 1e-8
        active = self._tt > eps
        # Sync once per training interval to validate signal quality.
        active_count = int(active.sum().detach().cpu().item())
        if active_count < int(self.cfg.min_active_params_for_error):
            return

        err2 = (a * a * self._uu - 2.0 * a * self._ut + self._tt) / (self._tt + eps)
        err2 = torch.where(active, err2, torch.zeros_like(err2))

        mean_err = torch.sqrt(err2[active].mean()).detach().float().cpu().item()
        self._errors.append(float(mean_err))

        if len(self._errors) >= int(self.cfg.error_window):
            avg = sum(self._errors) / len(self._errors)
            if avg > float(self.cfg.max_forecast_error):
                self._enabled = False
                return

        if self._enabled:
            loss = err2[active].mean()
            self._opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
            self._opt.step()

    def _compute_step_stats(self) -> None:
        # Fill uu,tt,ut,vv,uv per node on sketch (or full node tensor).
        for i, ns in enumerate(self._nodes):
            p = ns.tracked.param
            w = ns.view(p.data)
            wp = ns.view(ns.tracked.w_prev)
            u = ns.view(ns.tracked.delta_prev)
            g = ns.view(p.grad) if (p.grad is not None) else None

            uu, tt, ut, vv, uv = sketch_dot5(w, wp, u, g, ns.sketch_idx)
            self._uu[i] = uu
            self._tt[i] = tt
            self._ut[i] = ut
            self._vv[i] = vv
            self._uv[i] = uv

        # Numerical sanity
        self._uu.clamp_min_(0.0)
        self._vv.clamp_min_(0.0)
        self._tt.clamp_min_(0.0)

    def _commit_history_and_state(self) -> None:
        eps = 1e-8
        delta_rms = torch.sqrt(self._tt / self._sketch_len.clamp_min(1.0))
        grad_rms = torch.sqrt(self._vv / self._sketch_len.clamp_min(1.0))
        cos_dp = self._ut / torch.sqrt((self._uu * self._tt).clamp_min(eps))

        entry = torch.stack([delta_rms, grad_rms, cos_dp], dim=-1)

        self._hist[self._hist_i].copy_(entry)
        self._hist_i = (self._hist_i + 1) % self._hist.shape[0]
        self._hist_count = min(self._hist_count + 1, self._hist.shape[0])

        # Commit full (exact) delta_prev and w_prev for next step (per parameter tensor)
        with torch.no_grad():
            for tp in self._tracked_params:
                tp.delta_prev.copy_(tp.param.data).sub_(tp.w_prev)
                tp.w_prev.copy_(tp.param.data)

    def _apply_optimizer_state_policy(self, opt: torch.optim.Optimizer, policy: OptimizerStatePolicy) -> None:
        if policy == "keep":
            return

        tracked_params = {tp.param for tp in self._tracked_params}
        for group in opt.param_groups:
            for p in group.get("params", []):
                if p not in tracked_params:
                    continue
                st = opt.state.get(p, None)
                if not isinstance(st, dict):
                    continue
                for k, v in list(st.items()):
                    if torch.is_tensor(v):
                        with torch.no_grad():
                            v.zero_()
                    elif policy == "reset_all" and isinstance(v, (int, float)):
                        st[k] = 0

    def _advance_optimizer_steps_if_supported(self, opt: torch.optim.Optimizer, horizon: int) -> None:
        # Best-effort: advance per-parameter `step` counters (Adam/AdamW bias correction),
        # without inventing moment trajectories.
        h = int(horizon)
        if h <= 0:
            return
        tracked_params = {tp.param for tp in self._tracked_params}
        for group in opt.param_groups:
            for p in group.get("params", []):
                if p not in tracked_params:
                    continue
                st = opt.state.get(p, None)
                if not isinstance(st, dict):
                    continue
                if "step" not in st:
                    continue
                v = st["step"]
                if isinstance(v, int):
                    st["step"] = int(v + h)
                elif isinstance(v, float):
                    st["step"] = float(v + float(h))
                elif torch.is_tensor(v):
                    with torch.no_grad():
                        st["step"] = v + torch.as_tensor(h, device=v.device, dtype=v.dtype)

    def _sync_ema_model(self, ema_model: nn.Module) -> None:
        # Best-effort: copy model params to ema_model by name.
        src = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, p_ema in ema_model.named_parameters():
                p = src.get(name)
                if p is None:
                    continue
                p_ema.data.copy_(p.data)
