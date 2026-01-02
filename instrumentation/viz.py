"""Training visualization (viz) context for lightweight introspection.

We can't stream full activations/attention tensors during training: they're huge.
Instead we downsample aggressively and emit small, meaningful summaries that the
frontend can render in real-time (attention heatmaps, activation slices, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
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

    # For n-gram cache and other token-aware layers.
    input_ids: Tensor | None = None

    # Teacher-forced memory controls (Stage D1/D2).
    # Expected keys are implementation-defined (e.g., read_bucket/write_bucket/write_gate).
    mosaic_teacher: dict[str, Tensor] | None = None

    # If true, MOSAIC layers may record aux tensors for curriculum losses.
    mosaic_collect_aux: bool = False

    # Scheduled sampling probability: probability of using teacher controls when present.
    mosaic_teacher_p: float = 1.0

    # If true, MOSAIC layers may compute cheap scalar stats for logging.
    # This is independent of `enabled` (which controls heavy viz payloads).
    mosaic_stats_enabled: bool = False

    # Best-effort: filled by MOSAIC layers during forward.
    mosaic_aux_out: dict[str, Tensor] | None = None

    # Optional mask to drop local mixer contribution (forced-read dropout).
    # Expected shape: (B,T) or (T,) with values in {0,1}.
    mosaic_drop_local: Tensor | None = None

    # Accumulated MOSAIC memory stats for logging (filled by MOSAIC layers).
    mosaic_mem_stats: dict[str, float] = field(default_factory=dict)

