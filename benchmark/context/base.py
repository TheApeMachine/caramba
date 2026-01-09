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
from collector.measurement.context.result import ContextResult
from collector.measurement.context.sweep import ContextSweepMeasurement
from collector.measurement.context.decode import ContextDecodeMeasurement

from caramba.console.logger import Logger
logger: Logger = Logger()


def _sync(device: torch.device) -> None:
    """Synchronize device operations

    Waits for all pending operations on the given device to complete, ensuring
    accurate timing measurements. Supports both CUDA and MPS backends.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _load_prefix_tokens(dataset_path: str, n: int) -> list[int] | None:
    """Load prefix tokens from NumPy file

    Loads the first n token IDs from a NumPy array file, materializing only
    the needed prefix to minimize memory usage. Returns None if the file
    doesn't exist, can't be read, or doesn't have enough tokens.
    """
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
    """Parse KV cache kind from string

    Converts a string representation of a KV cache kind to the enum type,
    defaulting to FP16 if parsing fails. Handles case-insensitive matching
    and whitespace trimming for robustness.
    """
    s = str(x).strip().lower()
    try:
        return KVCacheKind(s)
    except Exception:
        return KVCacheKind.FP16


class BenchmarkContext:
    """Context length benchmark

    Measures model performance across different context lengths, including
    prefill time, decode throughput, and loss/perplexity. Uses chunked prefill
    to avoid materializing huge attention masks, and supports decode-at-context
    measurements to test generation performance after long contexts.
    """
    def __init__(self, config: ContextBenchmarkConfig, device: torch.device) -> None:
        """Initialize context benchmark

        Sets up the benchmark with configuration parameters and target device,
        preparing to run measurements across multiple context lengths.
        """
        self.config = config
        self.device = device

    def run(self, model: nn.Module, model_name: str) -> ContextResult:
        """Run context benchmark

        Executes the full benchmark suite across all configured context lengths,
        measuring prefill performance, single-token decode latency, and decode
        throughput. Uses chunked prefill to handle long contexts efficiently,
        and falls back to random tokens if no dataset is provided.
        """
        model.eval()
        out = ContextResult(model_name=str(model_name))

        # Best-effort vocab size detection for random token fallback.
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

            # Create (B, T) tensors for prefill; targets are shifted by 1 for next-token prediction.
            tok = torch.tensor(tokens, device=self.device, dtype=torch.long)
            x_all = tok[: int(ctx_len)].view(1, -1).repeat(bs, 1)
            y_all = tok[1 : int(ctx_len) + 1].view(1, -1).repeat(bs, 1)

            # Allocate KV caches for full sequence length (context + decode).
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

            # Prefill in chunks to avoid materializing huge attention masks.
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
                        # Keep attention mask size (t_q * t_k) under cap: t_k grows with seq_len.
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
                            # Compute loss on last chunk only for final position metrics.
                            loss_sum = F.cross_entropy(
                                last_logits.reshape(-1, last_logits.size(-1)),
                                y.reshape(-1),
                                reduction="sum",
                            )
                            last_loss = float(loss_sum.detach().float().cpu().item()) / float(max(1, y.numel()))
                            last_ppl = float(math.exp(last_loss)) if last_loss < 20 else float("inf")

                        pos += ch
            except Exception as e:
                logger.error(f"Error in context benchmark: {e}")
                ok = False

            t1 = time.perf_counter()

            # Measure single-token decode latency at full context length.
            decode_one_ms = float("nan")
            decode_one_tps = float("nan")
            if ok and last_logits is not None:
                try:
                    # Extract greedy next token from last position logits.
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

            # Decode throughput benchmark at context length (optional).
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

            # Warmup + timed decode using constant token to minimize sampling overhead.
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
