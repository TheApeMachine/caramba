"""Layer stats collector.

Collects lightweight activation statistics from attention layers.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any

from caramba.collector import Collector
from caramba.collector.layer.error import (
    LayerStatsError,
    LayerStatsErrorType,
)
from caramba.config.layer import AttentionLayerConfig


class LayerStatsCollector(Collector):
    """Layer stats collector."""

    def __init__(self, attn_modules: list[tuple[int, str, nn.Module]]) -> None:
        super().__init__(name="layer_stats")
        self.attn_modules = attn_modules
        self.enabled = False
        self.reset()

    def reset(self) -> None:
        self.counts: dict[int, int] = {}
        self.numel: dict[int, int] = {}
        self.sum_sq: dict[int, Tensor] = {}
        self.sum_abs: dict[int, Tensor] = {}
        self.max_abs: dict[int, Tensor] = {}
        self.shapes: dict[int, list[int]] = {}

    @torch.no_grad()
    def observe(self, idx: int, y: Tensor) -> None:
        """Observe a single layer output tensor."""
        try:
            z: Tensor = y.detach()
            if idx not in self.shapes:
                self.shapes[idx] = [int(x) for x in list(z.shape)]

            absz: Tensor = z.abs()
            # Use sum/amax (single-pass-ish reductions) and avoid .item() here
            sum_sq: Tensor = (z * z).sum()
            sum_abs: Tensor = absz.sum()
            max_abs: Tensor = absz.amax()
            n: int = z.numel()

            self.counts[idx] = self.counts.get(idx, 0) + 1
            self.numel[idx] = self.numel.get(idx, 0) + n

            if idx in self.sum_sq:
                self.sum_sq[idx] = self.sum_sq[idx] + sum_sq
                self.sum_abs[idx] = self.sum_abs[idx] + sum_abs
                self.max_abs[idx] = torch.maximum(self.max_abs[idx], max_abs)
            else:
                self.sum_sq[idx] = sum_sq
                self.sum_abs[idx] = sum_abs
                self.max_abs[idx] = max_abs
        except Exception as e:
            raise LayerStatsError(LayerStatsErrorType.LAYER_STATS_COLLECTION_FAILED, e) from e

    def finalize(self) -> list[dict[str, object]]:
        """Convert collected statistics to a serializable payload."""
        try:
            out: list[dict[str, object]] = []

            for idx, name, mod in self.attn_modules:
                n_int = int(self.numel.get(idx, 0))
                if n_int <= 0:
                    continue
                sum_sq_t = self.sum_sq.get(idx, None)
                sum_abs_t = self.sum_abs.get(idx, None)
                max_abs_t = self.max_abs.get(idx, None)
                if not isinstance(sum_sq_t, Tensor) or not isinstance(sum_abs_t, Tensor) or not isinstance(max_abs_t, Tensor):
                    continue

                # Move once; .item() now is off the critical path.
                sum_sq = sum_sq_t.float().cpu()
                sum_abs = sum_abs_t.float().cpu()
                max_abs = max_abs_t.float().cpu()
                n = float(n_int)

                ms = (sum_sq / n).item()
                ma = (sum_abs / n).item()
                mx = max_abs.item()

                rms = float(math.sqrt(max(0.0, ms)))

                cfg = getattr(mod, "config", None)
                cfg_attn = cfg if isinstance(cfg, AttentionLayerConfig) else None
                mode_obj: Any = getattr(mod, "mode", "")
                mode = str(getattr(mode_obj, "value", mode_obj))

                out.append({
                    "index": idx,
                    "name": name,
                    "type": "attention",
                    "shape": self.shapes.get(idx, None),
                    "mean_abs": ma,
                    "rms": rms,
                    "max_abs": mx,
                    "mode": mode,
                    "null_attn": bool(getattr(cfg_attn, "null_attn", False)),
                    "tie_qk": bool(getattr(cfg_attn, "tie_qk", False)),
                    "rope_semantic": bool(getattr(cfg_attn, "rope_semantic", False)),
                    "decoupled_gate": bool(getattr(cfg_attn, "decoupled_gate", False)),
                })
            return out
        except Exception as e:
            raise LayerStatsError(
                LayerStatsErrorType.LAYER_STATS_FINALIZATION_FAILED, e
            ) from e
