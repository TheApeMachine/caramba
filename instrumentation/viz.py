"""Training visualization (viz) context for lightweight introspection.

We can't stream full activations/attention tensors during training: they're huge.
Instead we downsample aggressively and emit small, meaningful summaries that the
frontend can render in real-time (attention heatmaps, activation slices, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from torch import Tensor


@dataclass(slots=True)
class TrainingVizContext:
    """Per-step collector for downsampled training telemetry.

    This is passed as `ctx=` through the model. Attention layers can detect it
    and record small samples.
    """
    enabled: bool
    step: int
    max_tokens: int = 16
    max_channels: int = 32
    max_heads: int = 4
    topk: int = 8

    # Internal buffers (index -> payload)
    layers: dict[int, dict[str, Any]] = field(default_factory=dict)

    def _layer(
        self, idx: int, *, name: str, mode: str, n_heads: int | None = None
    ) -> dict[str, Any]:
        d = self.layers.get(idx)
        if d is None:
            d = {"index": int(idx), "name": str(name), "mode": str(mode)}
            self.layers[idx] = d
        if n_heads is not None:
            try:
                d["n_heads"] = int(n_heads)
            except Exception:
                pass
        return d

    def record_activation_sample(
        self, *, idx: int, name: str, mode: str, n_heads: int | None = None, y: Tensor
    ) -> None:
        if not self.enabled:
            return
        try:
            # y: (B, T, D)
            if y.ndim != 3:
                return
            t = int(min(int(self.max_tokens), int(y.size(1))))
            c = int(min(int(self.max_channels), int(y.size(2))))
            sample = y[0, :t, :c].detach().float().cpu()
            self._layer(idx, name=name, mode=mode, n_heads=n_heads)["act"] = {
                "shape": [t, c],
                "values": sample.tolist(),
            }
        except Exception:
            return

    def record_attention_matrix(
        self,
        *,
        idx: int,
        name: str,
        mode: str,
        n_heads: int | None = None,
        matrices: list[Tensor],
        entropies: list[float] | None = None,
    ) -> None:
        """Record small attention heatmaps for a few heads.

        `matrices` should be a list of (tq, tk) tensors (already downsampled).
        """
        if not self.enabled:
            return
        try:
            mats = []
            for m in matrices[: int(self.max_heads)]:
                mm = m.detach().float().cpu()
                mats.append(mm.tolist())
            payload: dict[str, Any] = {"matrices": mats}
            if entropies is not None:
                payload["entropy"] = [float(x) for x in entropies[: int(self.max_heads)]]
            self._layer(idx, name=name, mode=mode, n_heads=n_heads)["attn"] = payload
        except Exception:
            return

    def to_event(self) -> dict[str, Any]:
        """Return JSONL-serializable event payload."""
        return {"layers": [self.layers[k] for k in sorted(self.layers.keys())]}


@dataclass(slots=True)
class TrainingVizMosaicContext(TrainingVizContext):
    """Training context that also carries MOSAIC control signals.

    This exists so we can pass one ctx object through the model that:
    - remains `isinstance(ctx, TrainingVizContext)` for attention viz hooks
    - carries extra fields needed for MOSAIC curriculum training (teacher actions, input_ids)
    """

    # Global RNG seed for deterministic dropout in attention layers.
    seed: int | None = None

    # For n-gram cache and other token-aware layers.
    input_ids: Tensor | None = None

    # Teacher-forced memory controls (Stage D1/D2).
    # Expected keys are implementation-defined (e.g., read_bucket/write_bucket/write_gate).
    memblock_teacher: dict[str, Tensor] | None = None

    # If true, MOSAIC layers may record aux tensors for curriculum losses.
    memblock_collect_aux: bool = False

    # Scheduled sampling probability: probability of using teacher controls when present.
    memblock_teacher_p: float = 1.0

    # If > 0, MOSAIC layers may disable writes for first N training steps.
    memblock_write_warmup_steps: int = 0

    # If true, MOSAIC layers may compute cheap scalar stats for logging.
    # This is independent of `enabled` (which controls heavy viz payloads).
    memblock_stats_enabled: bool = False

    # Filled by MOSAIC layers during forward.
    memblock_aux_out: dict[str, Tensor] | None = None

    # Optional mask to drop local mixer contribution (forced-read dropout).
    # Expected shape: (B,T) or (T,) with values in {0,1}.
    mosaic_drop_local: Tensor | None = None

    # Accumulated MOSAIC memory stats for logging (filled by MOSAIC layers).
    # Values may be Python floats or scalar tensors; logging code is responsible
    # for converting to JSON-safe floats in a batched way.
    memblock_mem_stats: dict[str, float | Tensor] = field(default_factory=dict)

    # Opaque state store for MOSAIC layers (e.g. n-gram cache state, persistent registers).
    # This must be in __slots__ for layers to attach state to the context.
    _mosaic: dict[str, Any] | None = None

    # Memory block state store (e.g. RMF, KV cache).
    # Must be in __slots__ for MemoryBlockStateStore to attach state.
    _memblock: dict[str, Any] | None = None
    
    # Training metrics for tuner optimization
    train_accuracy: float | None = None
    train_loss: float | None = None
    train_loss_variance: float | None = None
    _last_loss: float | None = None


@dataclass(slots=True)
class TrainingUAAContext(TrainingVizMosaicContext):
    """Training context for Utility-Aligned Attention (UAA).

    This context is used to request and hold differentiable attention probability
    slices from selected attention layers/heads for use in an auxiliary loss.

    Important:
    - Unlike `TrainingVizContext`, the stored tensors here are meant to stay on-device
      and participate in autograd (no CPU conversion / detaching).
    - This is separate from visualization to keep overhead bounded and explicit.
    """

    # Whether UAA capture/loss is enabled for this step/microbatch.
    uaa_enabled: bool = False

    # Which attention layer indices (by `_viz_index`) to capture from.
    # Convention: indices are assigned by the trainer's module discovery.
    uaa_layers: tuple[int, ...] = ()

    # Which head indices to capture (per-layer head indices, 0..n_heads-1).
    uaa_heads: tuple[int, ...] = ()

    # Which query position to align for (e.g. last token index).
    uaa_query_index: int = -1

    # Captured attention probabilities:
    #   layer_idx -> (B, H_sel, T_k) probabilities over key positions.
    uaa_attn: dict[int, Tensor] = field(default_factory=dict)

