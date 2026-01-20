"""Rotary positional embeddings (RoPE)

RoPE injects position information by rotating query/key features, which makes
attention scores depend on relative position and often generalizes better to
longer contexts than absolute embeddings.
"""
from __future__ import annotations

import logging
import math
import torch
from torch import nn
from collections.abc import Callable
from typing import TypeVar, cast

log = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., object])


def _dynamo_disable_typed(fn: _F) -> _F:
    """Typed wrapper for `torch._dynamo.disable`.

    Using `@torch._dynamo.disable` directly can cause some type checkers
    (basedpyright) to lose the decorated function signature, which then makes
    keyword calls (e.g. target_len=...) look invalid.
    """

    try:
        import torch._dynamo as _dynamo  # type: ignore

        return cast(_F, _dynamo.disable(fn))
    except Exception:
        return cast(_F, fn)


class RotaryEmbedding(nn.Module):
    """RoPE embedding cache

    Caching cos/sin tables is a practical acceleration: decoding would
    otherwise spend a surprising amount of time recomputing trigonometry for
    every token step.
    """

    inv_freq: torch.Tensor
    rot_dim: int

    def __init__(
        self,
        rot_dim: int,
        base: float = 10000.0,
        *,
        rope_scaling: dict[str, object] | None = None,
    ) -> None:
        """Initialize RoPE

        The `base` controls how quickly rotation frequencies decay across
        dimensions; changing it is one way to tune how position sensitivity is
        distributed across the head dimension.
        """
        super().__init__()
        if rot_dim % 2 != 0:
            raise ValueError(f"rot_dim must be even, got {rot_dim}")
        self.rot_dim = int(rot_dim)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rot_dim, 2, dtype=torch.float32)
                / float(self.rot_dim)
            )
        )
        # Optional scaling for Llama 3 ("llama3" RoPE) to match HF reference.
        # Implements the piecewise adjustment from transformers' modeling_rope_utils:
        # - long wavelengths (> low_freq_wavelen) are scaled by 1/factor (lower freq)
        # - short wavelengths (< high_freq_wavelen) are unchanged
        # - mid band is linearly interpolated between the two
        if rope_scaling:
            rt = str(rope_scaling.get("rope_type", rope_scaling.get("type", "")) or "").lower().strip()
            if rt == "llama3":
                try:
                    def _to_f(v: object, default: float) -> float:
                        try:
                            return float(v)  # type: ignore[arg-type]
                        except Exception:
                            return float(default)

                    def _to_i(v: object, default: int) -> int:
                        try:
                            return int(v)  # type: ignore[arg-type]
                        except Exception:
                            return int(default)

                    factor = _to_f(rope_scaling.get("factor", 8.0), 8.0)  # typical for llama3
                    low_f = _to_f(rope_scaling.get("low_freq_factor", 1.0), 1.0)
                    high_f = _to_f(rope_scaling.get("high_freq_factor", 4.0), 4.0)
                    old_ctx = _to_i(rope_scaling.get("original_max_position_embeddings", 8192), 8192)
                    if factor > 0 and low_f > 0 and high_f > 0 and old_ctx > 0:
                        low_wavelen = float(old_ctx) / float(low_f)
                        high_wavelen = float(old_ctx) / float(high_f)
                        # wavelen = 2*pi / inv_freq
                        wavelen = (2.0 * math.pi) / inv_freq
                        inv_scaled = inv_freq / float(factor)
                        # Smooth interpolation in the mid band.
                        denom = max(1e-12, (low_wavelen - high_wavelen))
                        smooth = (wavelen - high_wavelen) / float(denom)
                        smooth = smooth.clamp(0.0, 1.0)
                        inv_mid = (1.0 - smooth) * inv_freq + smooth * inv_scaled
                        inv_freq = torch.where(
                            wavelen > low_wavelen,
                            inv_scaled,
                            torch.where(wavelen < high_wavelen, inv_freq, inv_mid),
                        )
                except Exception as e:
                    raise ValueError(
                        "Invalid rope_scaling configuration. "
                        "Fix the manifest (rope_scaling.factor/low_freq_factor/high_freq_factor/"
                        "original_max_position_embeddings must be valid positive numbers)."
                    ) from e
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Tensor cache (preferred over Python dict for torch.compile compatibility).
        # These buffers are populated lazily on first use for the current device/dtype.
        self.register_buffer("_cos_cached", torch.empty((0, self.rot_dim // 2)), persistent=False)
        self.register_buffer("_sin_cached", torch.empty((0, self.rot_dim // 2)), persistent=False)
        self._cached_len: int = 0

    @staticmethod
    def _next_pow2(n: int) -> int:
        """Next power-of-two helper

        Power-of-two growth is a simple amortization trick that keeps cache
        expansions infrequent during long-context decoding.
        """
        n = int(n)
        if n <= 0:
            return 0
        return 1 << (n - 1).bit_length()

    @_dynamo_disable_typed
    def _ensure_cache(self, *, target_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """(Re)build cos/sin cache up to `target_len`.

        This runs outside torch.compile tracing to avoid Dynamo/SymPy issues.
        """
        target_len = int(target_len)
        if target_len <= int(self._cached_len):
            return

        start = int(self._cached_len)
        t = torch.arange(start, target_len, device=device, dtype=torch.float32)
        inv = self.inv_freq.to(device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv)
        cos_new = torch.cos(freqs).to(dtype=dtype)
        sin_new = torch.sin(freqs).to(dtype=dtype)

        if int(self._cached_len) == 0:
            self._cos_cached = cos_new
            self._sin_cached = sin_new
        else:
            self._cos_cached = torch.cat([self._cos_cached, cos_new], dim=0)
            self._sin_cached = torch.cat([self._sin_cached, sin_new], dim=0)
        self._cached_len = int(self._cos_cached.size(0))

    def _cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cached cos/sin tables.

        The cache grows geometrically so the common “one token at a time” decode
        path stays dominated by attention compute, not by cache maintenance.
        """
        seq_len_i = int(seq_len)

        # Ensure cache tensors live on the right device/dtype.
        # Note: buffers will typically follow the module, but this keeps correctness
        # for cases where rotate is called before `.to(device/dtype)`.
        if self._cos_cached.device != device or self._cos_cached.dtype != dtype:
            self._cos_cached = self._cos_cached.to(device=device, dtype=dtype)
            self._sin_cached = self._sin_cached.to(device=device, dtype=dtype)

        if int(self._cached_len) < seq_len_i:
            self._ensure_cache(target_len=self._next_pow2(seq_len_i), device=device, dtype=dtype)

        # Slice cached tables.
        return (self._cos_cached[:seq_len_i], self._sin_cached[:seq_len_i])

    def rotate(self, x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        """Apply rotary embedding rotation

        Only the first `rot_dim` features are rotated; the remaining dimensions
        pass through unchanged, which lets you reserve capacity for non-positional
        features if you want.
        """
        _B, _H, T, D = x.shape
        rot = self.rot_dim
        if rot > D:
            raise ValueError(f"rot_dim {rot} > head_dim {D}")

        cos, sin = self._cos_sin(pos_offset + T, x.device, x.dtype)
        cos = cos[pos_offset : pos_offset + T]
        sin = sin[pos_offset : pos_offset + T]

        from optimizer.kernels import rope_apply

        return rope_apply(x=x, cos=cos, sin=sin, rot_dim=int(self.rot_dim))
