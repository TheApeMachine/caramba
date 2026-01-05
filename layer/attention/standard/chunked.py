"""Chunked/windowed SDPA for standard attention.

This is used when:
- local_window is set, or
- q_chunk is set, or
- a KV cache is present (prefill/decode).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


class StandardSDPAChunked:
    """Scaled-dot-product attention with chunking/windowing for lower peak memory."""

    def run(
        self,
        *,
        qh: Tensor,
        kh: Tensor,
        vh: Tensor,
        is_causal: bool,
        scale: float | None,
        pos_offset: int,
        cache_pos: int | None,
        q_chunk: int,
        local_window: int | None,
        dropout_p: float,
        maybe_summarize_kv: Callable[..., tuple[Tensor, Tensor, Tensor]],
    ) -> Tensor:
        """Compute attention in query chunks and/or a sliding window.

        Args:
            qh/kh/vh: (B,H,T,hd)
            is_causal: causal masking
            scale: scalar scale for SDPA
            pos_offset: base offset when not cached
            cache_pos: cache.pos when cached, else None
            q_chunk: query chunk size
            local_window: local window size
            dropout_p: dropout probability
            maybe_summarize_kv: callable matching AttentionBase._maybe_summarize_kv
        """
        _B, _H, T, _D = qh.shape
        kT = int(kh.size(2))
        q_chunk = max(1, int(q_chunk))

        if cache_pos is not None:
            base_q = int(cache_pos) - int(T)
            q_pos_full = base_q + torch.arange(int(T), device=qh.device)
            k_pos_full = torch.arange(int(kT), device=qh.device)
        else:
            base_q = int(pos_offset)
            q_pos_full = base_q + torch.arange(int(T), device=qh.device)
            k_pos_full = int(pos_offset) + torch.arange(int(kT), device=qh.device)

        outs: list[Tensor] = []
        for i0 in range(0, int(T), int(q_chunk)):
            i1 = min(int(T), i0 + int(q_chunk))
            q_pos = q_pos_full[i0:i1]

            k0 = 0
            k1 = int(kT)
            if local_window is not None:
                w = int(local_window)
                if w > 0:
                    q_min = int(base_q + i0)
                    q_max = int(base_q + i1 - 1)
                    if bool(is_causal):
                        k0 = max(0, q_min - w + 1)
                        k1 = min(int(kT), q_max + 1)
                    else:
                        k0 = max(0, q_min - w + 1)
                        k1 = min(int(kT), q_max + w)

            q_slice = qh[:, :, i0:i1, :]
            k_slice = kh[:, :, k0:k1, :]
            v_slice = vh[:, :, k0:k1, :]

            if bool(is_causal) or local_window is not None:
                k_pos = k_pos_full[k0:k1]
                k_slice, v_slice, k_pos = maybe_summarize_kv(k=k_slice, v=v_slice, k_pos=k_pos)
                allowed = torch.ones((q_pos.numel(), k_pos.numel()), device=qh.device, dtype=torch.bool)
                if bool(is_causal):
                    allowed &= k_pos.view(1, -1) <= q_pos.view(-1, 1)
                if local_window is not None:
                    w = int(local_window)
                    if w > 0:
                        allowed &= k_pos.view(1, -1) >= (q_pos.view(-1, 1) - w + 1)
                        if not bool(is_causal):
                            allowed &= k_pos.view(1, -1) <= (q_pos.view(-1, 1) + w - 1)
                attn_mask = allowed
            else:
                k_pos = k_pos_full[k0:k1]
                k_slice, v_slice, _ = maybe_summarize_kv(k=k_slice, v=v_slice, k_pos=k_pos)
                attn_mask = None

            out = F.scaled_dot_product_attention(
                q_slice,
                k_slice,
                v_slice,
                attn_mask=attn_mask,
                dropout_p=float(dropout_p),
                is_causal=False,
                scale=scale,
            )
            outs.append(out)

        return torch.cat(outs, dim=2)

