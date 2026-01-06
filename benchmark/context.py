"""Long-context benchmark (chunked prefill + decode-at-context).

This is a manifest-driven replacement for the useful parts of the old
eval_ckpt.py, implemented in the Caramba benchmarking framework.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from caramba.config.benchmark import ContextBenchmarkConfig
from caramba.config.kvcache import KVCacheKind
from caramba.infer.context import InferContext
from caramba.infer.generate import create_caches


@dataclass
class ContextSweepMeasurement:
    context_len: int
    chunk_size_used: int
    batch_size: int
    prefill_total_s: float
    prefill_last_chunk_ms: float
    decode_one_ms: float
    decode_one_tok_per_s: float
    loss_last_chunk: float
    ppl_last_chunk: float
    ok: bool


@dataclass
class ContextDecodeMeasurement:
    context_len: int
    chunk_size_used: int
    batch_size: int
    decode_len: int
    decode_warmup: int
    prefill_total_s: float
    decode_total_ms: float
    decode_tok_per_s: float
    ok: bool


@dataclass
class ContextResult:
    model_name: str
    sweep: list[ContextSweepMeasurement] = field(default_factory=list)
    decode: list[ContextDecodeMeasurement] = field(default_factory=list)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _load_prefix_tokens(dataset_path: str, n: int) -> list[int] | None:
    """Load a prefix of token IDs from a .npy file."""
    try:
        import numpy as np  # type: ignore[import-not-found]

        arr = np.load(str(dataset_path), mmap_mode="r").reshape(-1)
        if int(arr.shape[0]) < int(n):
            return None
        # Materialize only the needed prefix.
        xs = arr[: int(n)].astype("int64", copy=True)
        return [int(x) for x in xs.tolist()]
    except Exception:
        return None


def _kv_kind(x: str) -> KVCacheKind:
    s = str(x).strip().lower()
    try:
        return KVCacheKind(s)
    except Exception:
        return KVCacheKind.FP16


class ContextBenchmark:
    def __init__(self, config: ContextBenchmarkConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

    def run(self, model: nn.Module, model_name: str) -> ContextResult:
        model.eval()
        out = ContextResult(model_name=str(model_name))

        # Best-effort vocab size for random fallback.
        vocab_size = getattr(model, "vocab_size", None)
        if vocab_size is None:
            cfg = getattr(model, "config", None)
            vocab_size = getattr(cfg, "vocab_size", None) if cfg is not None else None
        if vocab_size is None or int(vocab_size) <= 0:
            vocab_size = 50304

        bs = int(self.config.batch_size)
        chunk_req = int(self.config.chunk_size)
        max_mask = int(self.config.max_mask_elems)
        decode_len = int(self.config.decode_len)
        decode_warmup = int(self.config.decode_warmup)

        for ctx_len in [int(x) for x in self.config.context_lengths]:
            # Need ctx_len + 1 for next-token targets and decode-at-context.
            need = int(ctx_len) + 1
            tokens: list[int] | None = None
            if self.config.dataset:
                tokens = _load_prefix_tokens(str(self.config.dataset), need)
            if tokens is None:
                g = torch.Generator(device=self.device).manual_seed(1337)
                t = torch.randint(0, int(vocab_size), (need,), generator=g, device=self.device, dtype=torch.long)
                tokens = [int(x) for x in t.cpu().tolist()]

            # (B, T) for prefill; targets are shifted by 1.
            tok = torch.tensor(tokens, device=self.device, dtype=torch.long)
            x_all = tok[: int(ctx_len)].view(1, -1).repeat(bs, 1)
            y_all = tok[1 : int(ctx_len) + 1].view(1, -1).repeat(bs, 1)

            # Allocate caches for full length (context + decode).
            caches = create_caches(
                model,
                batch_size=bs,
                max_seq_len=int(ctx_len) + int(decode_len) + 2,
                device=self.device,
                cache_kind=_kv_kind(str(self.config.cache_kind)),
                cache_qblock=32,
                cache_residual_len=0,
            )
            ictx = InferContext(caches=caches)

            # Prefill in chunks (avoid huge mask materialization).
            pos = 0
            last_chunk_ms = 0.0
            last_loss = float("nan")
            last_ppl = float("inf")
            ok = True
            last_logits: Tensor | None = None
            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    while pos < int(ctx_len):
                        remaining = int(ctx_len - pos)
                        seq_len = int(pos + min(chunk_req, remaining))
                        # Keep (t_q * t_k) under a cap: t_k roughly grows with seq_len.
                        t_allowed = max(1, int(max_mask // max(1, seq_len)))
                        ch = int(min(chunk_req, remaining, t_allowed))

                        x = x_all[:, pos : pos + ch]
                        y = y_all[:, pos : pos + ch]
                        ictx.begin(pos_offset=int(pos))
                        tcs = time.perf_counter()
                        logits = model(x, ctx=ictx)  # type: ignore[call-arg]
                        ictx.ensure_consumed()
                        _sync(self.device)
                        tce = time.perf_counter()

                        if pos + ch >= int(ctx_len):
                            last_chunk_ms = float((tce - tcs) * 1000.0)
                            last_logits = cast(Tensor, logits)
                            # loss on last chunk only
                            loss_sum = F.cross_entropy(
                                last_logits.reshape(-1, last_logits.size(-1)),
                                y.reshape(-1),
                                reduction="sum",
                            )
                            last_loss = float(loss_sum.detach().float().cpu().item()) / float(max(1, y.numel()))
                            last_ppl = float(math.exp(last_loss)) if last_loss < 20 else float("inf")

                        pos += ch
            except Exception:
                ok = False
            t1 = time.perf_counter()

            # Decode one token at full context.
            decode_one_ms = float("nan")
            decode_one_tps = float("nan")
            if ok and last_logits is not None:
                try:
                    # greedy token from last position
                    nxt = torch.argmax(last_logits[:, -1, :], dim=-1, keepdim=True)
                    ictx.begin(pos_offset=int(ctx_len))
                    tds = time.perf_counter()
                    _ = model(nxt, ctx=ictx)  # type: ignore[call-arg]
                    ictx.ensure_consumed()
                    _sync(self.device)
                    tde = time.perf_counter()
                    decode_one_ms = float((tde - tds) * 1000.0)
                    decode_one_tps = float(bs / max(1e-9, float(tde - tds)))
                except Exception:
                    ok = False

            out.sweep.append(
                ContextSweepMeasurement(
                    context_len=int(ctx_len),
                    chunk_size_used=int(chunk_req),
                    batch_size=int(bs),
                    prefill_total_s=float(t1 - t0),
                    prefill_last_chunk_ms=float(last_chunk_ms),
                    decode_one_ms=float(decode_one_ms),
                    decode_one_tok_per_s=float(decode_one_tps),
                    loss_last_chunk=float(last_loss),
                    ppl_last_chunk=float(last_ppl),
                    ok=bool(ok),
                )
            )

            # Decode throughput benchmark at context (optional).
            if not ok:
                out.decode.append(
                    ContextDecodeMeasurement(
                        context_len=int(ctx_len),
                        chunk_size_used=int(chunk_req),
                        batch_size=int(bs),
                        decode_len=int(decode_len),
                        decode_warmup=int(decode_warmup),
                        prefill_total_s=float(t1 - t0),
                        decode_total_ms=float("nan"),
                        decode_tok_per_s=float("nan"),
                        ok=False,
                    )
                )
                continue

            # Warmup + timed decode using a constant token to reduce model-dependent sampling overhead.
            tok0 = torch.zeros((bs, 1), device=self.device, dtype=torch.long)
            try:
                with torch.no_grad():
                    for i in range(int(decode_warmup)):
                        ictx.begin(pos_offset=int(ctx_len) + i)
                        _ = model(tok0, ctx=ictx)  # type: ignore[call-arg]
                        ictx.ensure_consumed()
                _sync(self.device)

                tds = time.perf_counter()
                with torch.no_grad():
                    for i in range(int(decode_len)):
                        ictx.begin(pos_offset=int(ctx_len) + int(decode_warmup) + i)
                        _ = model(tok0, ctx=ictx)  # type: ignore[call-arg]
                        ictx.ensure_consumed()
                _sync(self.device)
                tde = time.perf_counter()

                dt_ms = float((tde - tds) * 1000.0)
                tps = float((bs * int(decode_len)) / max(1e-9, float(tde - tds)))
                out.decode.append(
                    ContextDecodeMeasurement(
                        context_len=int(ctx_len),
                        chunk_size_used=int(chunk_req),
                        batch_size=int(bs),
                        decode_len=int(decode_len),
                        decode_warmup=int(decode_warmup),
                        prefill_total_s=float(t1 - t0),
                        decode_total_ms=float(dt_ms),
                        decode_tok_per_s=float(tps),
                        ok=True,
                    )
                )
            except Exception:
                out.decode.append(
                    ContextDecodeMeasurement(
                        context_len=int(ctx_len),
                        chunk_size_used=int(chunk_req),
                        batch_size=int(bs),
                        decode_len=int(decode_len),
                        decode_warmup=int(decode_warmup),
                        prefill_total_s=float(t1 - t0),
                        decode_total_ms=float("nan"),
                        decode_tok_per_s=float("nan"),
                        ok=False,
                    )
                )

        return out

