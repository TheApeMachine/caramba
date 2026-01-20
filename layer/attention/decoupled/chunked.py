"""Chunked/windowed DBA attention

This module computes decoupled attention in smaller query chunks (and optionally
with a local window), which keeps memory usage bounded when sequences get long.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from carmath import neg_inf


class DecoupledSDPAChunked:
    """Chunked DBA attention helper

    Chunking is especially useful for DBA because the layer builds multiple
    score matrices (semantic and geometric); slicing reduces the size of those
    intermediates.
    """

    def run(
        self,
        *,
        qsh: Tensor,
        ksh: Tensor,
        qgh: Tensor,
        kgh: Tensor,
        vh: Tensor,
        is_causal: bool,
        mask: Tensor | None,
        cache_pos: int | None,
        q_chunk: int,
        local_window: int | None,
        sem_scale: float,
        geo_scale: float,
        dropout_p: float,
        null_enabled: bool,
        null_kv: Callable[..., tuple[Tensor, Tensor, Tensor]],
        maybe_summarize_decoupled: Callable[..., tuple[Tensor, Tensor, Tensor, Tensor]],
    ) -> Tensor:
        """Compute DBA attention in chunks/windows

        The semantic and geometric paths are combined either via SDPA over a
        concatenated representation or via explicit score matrices when a mask
        is provided.
        """
        B, _H, T, _ = qsh.shape
        kT = int(ksh.size(2))
        q_chunk = max(1, int(q_chunk))
        ninfty = neg_inf(qsh.dtype)

        if cache_pos is not None:
            base_q = int(cache_pos) - int(T)
            q_pos_full = base_q + torch.arange(int(T), device=qsh.device)
            k_pos_full = torch.arange(int(kT), device=qsh.device)
        else:
            base_q = 0
            q_pos_full = torch.arange(int(T), device=qsh.device)
            k_pos_full = torch.arange(int(kT), device=qsh.device)

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

            k_pos = k_pos_full[k0:k1]
            q_slice_sem = qsh[:, :, i0:i1, :]
            q_slice_geo = qgh[:, :, i0:i1, :]
            k_slice_sem = ksh[:, :, k0:k1, :]
            k_slice_geo = kgh[:, :, k0:k1, :]
            v_slice = vh[:, :, k0:k1, :]

            k_slice_sem, k_slice_geo, v_slice, k_pos = maybe_summarize_decoupled(
                k_sem=k_slice_sem,
                k_geo=k_slice_geo,
                v=v_slice,
                k_pos=k_pos,
            )

            if null_enabled:
                ksn, kgn, vn = null_kv(B=B, dtype=qsh.dtype, device=qsh.device)
                k_slice_sem = torch.cat([ksn, k_slice_sem], dim=2)
                k_slice_geo = torch.cat([kgn, k_slice_geo], dim=2)
                v_slice = torch.cat([vn.to(dtype=v_slice.dtype), v_slice], dim=2)

            if mask is None:
                attn_mask = None
                if bool(is_causal) or local_window is not None:
                    keep_tokens = torch.ones((q_pos.numel(), k_pos.numel()), device=qsh.device, dtype=torch.bool)
                    if bool(is_causal):
                        keep_tokens &= k_pos.view(1, -1) <= q_pos.view(-1, 1)
                    if local_window is not None:
                        w = int(local_window)
                        if w > 0:
                            keep_tokens &= k_pos.view(1, -1) >= (q_pos.view(-1, 1) - w + 1)
                            if not bool(is_causal):
                                keep_tokens &= k_pos.view(1, -1) <= (q_pos.view(-1, 1) + w - 1)
                    if null_enabled:
                        keep_null = torch.ones((q_pos.numel(), 1), device=qsh.device, dtype=torch.bool)
                        keep = torch.cat([keep_null, keep_tokens], dim=1)
                    else:
                        keep = keep_tokens
                    attn_mask = keep

                q_cat = torch.cat([q_slice_sem * float(sem_scale), q_slice_geo * float(geo_scale)], dim=-1)
                k_cat = torch.cat([k_slice_sem, k_slice_geo], dim=-1)
                q_cat, k_cat = self._pad_qk_to_v_dim(q_cat, k_cat, v_slice)
                out = F.scaled_dot_product_attention(
                    q_cat,
                    k_cat,
                    v_slice,
                    attn_mask=attn_mask,
                    dropout_p=float(dropout_p),
                    is_causal=False,
                    scale=1.0,
                )
            else:
                sem_scores = torch.matmul(q_slice_sem, k_slice_sem.transpose(-2, -1)) * float(sem_scale)
                geo_scores = torch.matmul(q_slice_geo, k_slice_geo.transpose(-2, -1)) * float(geo_scale)
                scores = sem_scores + geo_scores
                m = mask[..., i0:i1, k0:k1]
                if null_enabled:
                    keep_null = torch.ones((*m.shape[:-1], 1), device=m.device, dtype=torch.bool)
                    m = torch.cat([keep_null, m], dim=-1)
                scores = scores.masked_fill(~m, ninfty)
                attn = F.softmax(scores.float(), dim=-1).to(qsh.dtype)
                out = torch.matmul(attn, v_slice)
            outs.append(out)

        return torch.cat(outs, dim=2)

    @staticmethod
    def _pad_qk_to_v_dim(q_cat: Tensor, k_cat: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Pad Q/K head dim up to V head dim when needed.

        Some SDPA backends are sensitive to Q/K/V head-dim mismatches. DBA often
        uses smaller Q/K dims (sem+geo) than V dims (attn_dim). Zero-padding Q/K
        preserves the attention math while keeping kernels shape-safe.
        """
        qk = int(q_cat.size(-1))
        vd = int(v.size(-1))
        if qk == vd:
            return q_cat, k_cat
        if qk > vd:
            raise RuntimeError(
                f"DBA Q/K head dim ({qk}) must be <= V head dim ({vd}). "
                "Decrease sem_dim+geo_dim or increase attn_dim."
            )
        pad = vd - qk
        return F.pad(q_cat, (0, pad)), F.pad(k_cat, (0, pad))

