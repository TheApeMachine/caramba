"""Prepare a tokenized FineWeb-style dataset as a 1D .npy array.

This script exists to avoid "teacher is jacked" failures caused by tokenizer/data
mismatch. For Llama teachers, you MUST tokenize with the matching Llama tokenizer.

Example:
  python3 prepare_fineweb.py --tokens 100M --output fineweb_100m.npy \\
    --tokenizer llama --model-id meta-llama/Llama-3.2-1B
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def _parse_tokens(s: str) -> int:
    t = str(s).strip().lower().replace("_", "")
    mult = 1
    if t.endswith("k"):
        mult = 1_000
        t = t[:-1]
    elif t.endswith("m"):
        mult = 1_000_000
        t = t[:-1]
    elif t.endswith("b"):
        mult = 1_000_000_000
        t = t[:-1]
    n = int(float(t) * mult)
    if n <= 0:
        raise ValueError("tokens must be > 0")
    return n


def _iter_text(ds) -> Iterable[str]:
    # FineWeb-like datasets commonly use "text". Allow overrides via --text-field.
    for ex in ds:
        if not isinstance(ex, dict):
            continue
        txt = ex.get("text", None)
        if isinstance(txt, str) and txt:
            yield txt


def prepare_fineweb_npy(
    *,
    tokens: str,
    output: str | Path,
    tokenizer: str = "llama",
    model_id: str = "meta-llama/Llama-3.2-1B",
    dataset: str = "HuggingFaceFW/fineweb",
    subset: str | None = None,
    split: str = "train",
    text_field: str = "text",
    max_chars: int = 50_000,
    append_eos: bool = True,
    append_bos: bool = False,
    overwrite: bool = False,
) -> Path:
    """Stream a HF dataset and write a 1D int32 token array to .npy."""
    target_tokens = _parse_tokens(tokens)
    out_path = Path(output)
    if out_path.suffix.lower() != ".npy":
        raise ValueError("output must end with .npy")
    if out_path.exists() and not bool(overwrite):
        return out_path

    # Imports are intentionally inside the function so the package can be imported without deps.
    import numpy as np

    tok_kind = str(tokenizer).strip().lower()
    if tok_kind == "llama":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(model_id), use_fast=True, trust_remote_code=False)
        bos_id = getattr(tok, "bos_token_id", None)
        eos_id = getattr(tok, "eos_token_id", None)

        def encode(text: str) -> list[int]:
            return list(tok.encode(text, add_special_tokens=False))

    elif tok_kind == "tiktoken":
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        def encode(text: str) -> list[int]:
            return list(enc.encode(text))
        bos_id = None
        eos_id = None

    else:
        raise ValueError(f"Unsupported tokenizer={tokenizer!r} (expected 'llama' or 'tiktoken')")

    from datasets import load_dataset

    ds = load_dataset(str(dataset), subset, split=str(split), streaming=True)

    buf: list[int] = []
    n = 0
    report_every = max(250_000, target_tokens // 100)

    for ex in ds:
        if not isinstance(ex, dict):
            continue
        raw = ex.get(str(text_field), None)
        if not isinstance(raw, str) or not raw:
            continue
        txt = raw[: int(max_chars)]
        ids = encode(txt)
        if not ids:
            continue
        if append_bos and bos_id is not None:
            buf.append(int(bos_id))
        buf.extend(ids)
        if append_eos and eos_id is not None:
            buf.append(int(eos_id))
        n = len(buf)
        if n >= target_tokens:
            break
        if n // report_every != (n - len(ids)) // report_every:
            pct = 100.0 * float(n) / float(target_tokens)
            print(f"[prepare_fineweb] tokens={n:,}/{target_tokens:,} ({pct:.1f}%)")

    if len(buf) < target_tokens:
        raise RuntimeError(f"Dataset exhausted early: got {len(buf):,} tokens, expected {target_tokens:,}")

    arr = np.asarray(buf[:target_tokens], dtype=np.int32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), arr)
    mb = arr.nbytes / (1024 * 1024)
    print(f"[prepare_fineweb] wrote {out_path} ({arr.shape[0]:,} tokens, {mb:.1f} MB)")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Tokenize FineWeb to .npy tokens")
    ap.add_argument("--tokens", type=str, required=True, help="How many tokens to emit (e.g. 100M, 1B).")
    ap.add_argument("--output", type=str, required=True, help="Output .npy path (1D int32).")
    ap.add_argument(
        "--tokenizer",
        type=str,
        default="llama",
        choices=["llama", "tiktoken"],
        help="Tokenizer backend to use.",
    )
    ap.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace tokenizer model id (for --tokenizer llama).",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb",
        help="HuggingFace dataset id to stream.",
    )
    ap.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional dataset subset/config name (HF datasets 'name' argument).",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train).",
    )
    ap.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Field containing text samples (default: text).",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=50_000,
        help="Truncate very long samples for stability.",
    )
    ap.add_argument(
        "--append-eos",
        action="store_true",
        default=True,
        help="Append EOS token between samples when available (default: true).",
    )
    ap.add_argument(
        "--no-append-eos",
        action="store_true",
        default=False,
        help="Disable EOS insertion (not recommended for Llama teachers).",
    )
    ap.add_argument(
        "--append-bos",
        action="store_true",
        default=False,
        help="Prepend BOS token before each sample when available (default: false).",
    )
    args = ap.parse_args(argv)
    _ = prepare_fineweb_npy(
        tokens=str(args.tokens),
        output=str(args.output),
        tokenizer=str(args.tokenizer),
        model_id=str(args.model_id),
        dataset=str(args.dataset),
        subset=(str(args.subset) if args.subset is not None else None),
        split=str(args.split),
        text_field=str(args.text_field),
        max_chars=int(args.max_chars),
        append_eos=bool(args.append_eos) and not bool(args.no_append_eos),
        append_bos=bool(args.append_bos),
        overwrite=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

