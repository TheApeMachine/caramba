"""Chunked linear + cross entropy loss.

This module implements a memory-efficient alternative to computing next-token
cross entropy from full logits.

What it is:
- A custom autograd function that computes CE loss for logits = X @ W^T (+bias)
  without materializing the full (N, V) logits matrix in memory.

Why it is:
- For large vocabularies (e.g. 50k) and long sequences (e.g. 2048), the logits
  tensor (B*T*V) can be multiple GiB and can dominate both memory and runtime.
- Chunking over the vocabulary dimension allows larger microbatches to fit and
  reduces allocator pressure, which often improves end-to-end throughput.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


def _require(cond: bool, *, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


@dataclass(frozen=True, slots=True)
class ChunkedCELossConfig:
    """Configuration for chunked CE.

    vocab_chunk: number of vocab rows per chunk.
    ignore_index: label value to ignore.
    """

    vocab_chunk: int = 8192
    ignore_index: int = -100


class _ChunkedLinearCEFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: Tensor,  # (N, D)
        w_t: Tensor,  # (D, V) (transposed weight)
        target: Tensor,  # (N,)
        bias: Optional[Tensor],
        vocab_chunk: int,
        ignore_index: int,
    ) -> Tensor:
        _require(x.ndim == 2, msg=f"x must be (N,D), got {tuple(x.shape)}")
        _require(w_t.ndim == 2, msg=f"w_t must be (D,V), got {tuple(w_t.shape)}")
        _require(target.ndim == 1, msg=f"target must be (N,), got {tuple(target.shape)}")
        _require(int(x.size(0)) == int(target.size(0)), msg="x/target N mismatch")
        _require(int(w_t.size(0)) == int(x.size(1)), msg="w_t D mismatch")
        if bias is not None:
            _require(bias.ndim == 1 and int(bias.numel()) == int(w_t.size(1)), msg="bias must be (V,)")
        _require(int(vocab_chunk) > 0, msg="vocab_chunk must be > 0")

        device = x.device
        dtype = x.dtype
        V = int(w_t.size(1))
        N = int(x.size(0))

        tgt = target.to(device=device, dtype=torch.long)
        valid = tgt != int(ignore_index)

        # Use fp32 for stability/perf, but preserve fp64 for gradcheck correctness.
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32

        # Streaming log-sum-exp across vocab chunks.
        m = torch.full((N,), -float("inf"), device=device, dtype=compute_dtype)
        s = torch.zeros((N,), device=device, dtype=compute_dtype)
        tlogit = torch.zeros((N,), device=device, dtype=compute_dtype)

        # Compute LSE in fp32 for stability; matmul uses tensor cores for bf16/fp16.
        for v0 in range(0, V, int(vocab_chunk)):
            v1 = min(V, v0 + int(vocab_chunk))
            w_chunk_t = w_t[:, v0:v1]  # (D, C)
            logits = (x @ w_chunk_t).to(compute_dtype)  # (N, C)
            if bias is not None:
                logits = logits + bias[v0:v1].to(compute_dtype).view(1, -1)

            m_chunk = logits.max(dim=1).values
            m_new = torch.maximum(m, m_chunk)
            # rescale running sum
            s = s * torch.exp(m - m_new) + torch.exp(logits - m_new.view(-1, 1)).sum(dim=1)
            m = m_new

            # Capture target logit if the target falls in this chunk.
            in_chunk = valid & (tgt >= v0) & (tgt < v1)
            if bool(in_chunk.any().item()):
                idx = (tgt[in_chunk] - v0).to(torch.long)
                tlogit[in_chunk] = logits[in_chunk, idx]

        lse = m + torch.log(s.clamp_min(1e-20))
        # Loss per token: lse - logit[target]
        loss_vec = (lse - tlogit)
        loss_vec = torch.where(valid, loss_vec, torch.zeros_like(loss_vec))
        denom = valid.detach().float().sum().clamp_min(1.0)
        loss = loss_vec.sum() / denom

        ctx.save_for_backward(x, w_t, tgt, valid, bias if bias is not None else torch.empty(0, device=device))
        ctx.meta = (int(vocab_chunk), int(ignore_index), V)
        return loss.to(dtype=compute_dtype)

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        x, w_t, tgt, valid, bias_saved = ctx.saved_tensors
        vocab_chunk, ignore_index, V = ctx.meta  # type: ignore[attr-defined]
        bias = None if int(bias_saved.numel()) == 0 else bias_saved

        N, D = int(x.size(0)), int(x.size(1))
        _ = ignore_index

        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32

        # Recompute LSE in compute_dtype (fp32 fast path, fp64 for gradcheck).
        m = torch.full((N,), -float("inf"), device=x.device, dtype=compute_dtype)
        s = torch.zeros((N,), device=x.device, dtype=compute_dtype)
        for v0 in range(0, int(V), int(vocab_chunk)):
            v1 = min(int(V), v0 + int(vocab_chunk))
            logits = (x @ w_t[:, v0:v1]).to(compute_dtype)
            if bias is not None:
                logits = logits + bias[v0:v1].to(compute_dtype).view(1, -1)
            m_chunk = logits.max(dim=1).values
            m_new = torch.maximum(m, m_chunk)
            s = s * torch.exp(m - m_new) + torch.exp(logits - m_new.view(-1, 1)).sum(dim=1)
            m = m_new
        lse = m + torch.log(s.clamp_min(1e-20))

        denom = valid.detach().to(dtype=compute_dtype).sum().clamp_min(1.0)
        g = grad_out.to(compute_dtype) / denom

        dx = torch.zeros_like(x, dtype=compute_dtype)
        dw_t = torch.zeros_like(w_t, dtype=compute_dtype)
        db = torch.zeros((int(V),), device=x.device, dtype=compute_dtype) if bias is not None else None

        for v0 in range(0, int(V), int(vocab_chunk)):
            v1 = min(int(V), v0 + int(vocab_chunk))
            w_chunk_t = w_t[:, v0:v1]  # (D, C)
            logits = (x @ w_chunk_t).to(compute_dtype)  # (N, C)
            if bias is not None:
                logits = logits + bias[v0:v1].to(compute_dtype).view(1, -1)

            p = torch.exp(logits - lse.view(-1, 1))  # (N, C)
            p = torch.where(valid.view(-1, 1), p, torch.zeros_like(p))

            # Subtract 1 for target class within this chunk.
            in_chunk = valid & (tgt >= v0) & (tgt < v1)
            if bool(in_chunk.any().item()):
                idx = (tgt[in_chunk] - v0).to(torch.long)
                p[in_chunk, idx] = p[in_chunk, idx] - 1.0

            dlogits = p * g  # (N, C)
            # dx += dlogits @ W_chunk
            dx = dx + (dlogits @ w_chunk_t.t().to(compute_dtype))
            # dW_chunk += dlogits^T @ x
            dw_t[:, v0:v1] = dw_t[:, v0:v1] + (x.to(compute_dtype).t() @ dlogits)
            if db is not None:
                db[v0:v1] = db[v0:v1] + dlogits.sum(dim=0)

        dx = dx.to(dtype=x.dtype)
        dw_t = dw_t.to(dtype=w_t.dtype)
        db_out = db.to(dtype=bias.dtype) if (db is not None and bias is not None) else None

        # Return grads for (x, w_t, target, bias, vocab_chunk, ignore_index)
        return dx, dw_t, None, db_out, None, None


def chunked_linear_cross_entropy(
    *,
    x: Tensor,
    weight: Tensor,  # (V, D)
    target: Tensor,
    bias: Tensor | None = None,
    cfg: ChunkedCELossConfig | None = None,
) -> Tensor:
    """Compute CE for logits = x @ weight^T (+bias) without materializing logits.

    x: (N, D)
    weight: (V, D)
    target: (N,)
    """
    c = cfg or ChunkedCELossConfig()
    w_t = weight.t().contiguous()
    out = _ChunkedLinearCEFn.apply(x, w_t, target, bias, int(c.vocab_chunk), int(c.ignore_index))
    if not isinstance(out, Tensor):
        raise TypeError("chunked_linear_cross_entropy must return a Tensor")
    return out

