"""FineWeb preparation

Builds tokenized `.npy` datasets from HuggingFace `datasets` sources.

Why this exists:
- Training should not tokenize raw text on-the-fly.
- Manifests should be able to request dataset preparation deterministically.
- The output format must be compatible with `caramba.data.npy.NpyDataset`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


class TextTokenizer(Protocol):
    """Tokenizer protocol for dataset preparation."""

    def encode(self, text: str) -> list[int]: ...

    @property
    def bos_token_id(self) -> int | None: ...

    @property
    def eos_token_id(self) -> int | None: ...


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """Token budget parsed from manifest strings like '100M'."""

    tokens: int


@dataclass(slots=True)
class FinewebNpyPreparer:
    """FineWeb-to-NPY preparer.

    Converts a streamed HuggingFace dataset split into a fixed-length token array
    stored as `.npy` via a memory-mapped writer.
    """

    def run(
        self,
        *,
        tokens: str,
        output: str,
        tokenizer: str,
        model_id: str,
        dataset: str,
        subset: str | None,
        split: str,
        text_field: str,
        max_chars: int,
        append_eos: bool,
        append_bos: bool,
        overwrite: bool,
    ) -> None:
        out_path = self.validate_output_path(output, overwrite=bool(overwrite))
        budget = self.parse_budget(tokens)
        tok = self.build_tokenizer(kind=str(tokenizer), model_id=str(model_id), append_bos=bool(append_bos), append_eos=bool(append_eos))
        ds = self.load_stream(dataset=str(dataset), subset=subset, split=str(split))
        mm = self.open_memmap(path=out_path, n_tokens=int(budget.tokens))
        meta = self.write_tokens(
            mm=mm,
            ds=ds,
            tokenizer=tok,
            text_field=str(text_field),
            max_chars=int(max_chars),
            append_bos=bool(append_bos),
            append_eos=bool(append_eos),
        )
        self.write_meta(out_path=out_path, meta=meta)

    def validate_output_path(self, output: str, *, overwrite: bool) -> Path:
        p = Path(str(output))
        if not str(p):
            raise ValueError("output must be non-empty")
        if p.suffix.lower() != ".npy":
            raise ValueError(f"output must end with .npy, got {p}")
        if p.exists() and not bool(overwrite):
            raise FileExistsError(f"Output exists; set overwrite=true to rebuild: {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and bool(overwrite):
            p.unlink()
        return p

    def parse_budget(self, tokens: str) -> TokenBudget:
        s = str(tokens).strip()
        m = re.fullmatch(r"(\\d+)([KMB])?", s, flags=re.IGNORECASE)
        if m is None:
            raise ValueError(f"Invalid token budget {tokens!r}. Use forms like '100M', '500K', '1000000'.")
        n = int(m.group(1))
        suf = (m.group(2) or "").upper()
        mult = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suf]
        total = int(n * mult)
        if total < 2:
            raise ValueError("Token budget must be >= 2")
        return TokenBudget(tokens=total)

    def build_tokenizer(self, *, kind: str, model_id: str, append_bos: bool, append_eos: bool) -> TextTokenizer:
        k = str(kind).strip().lower()
        if k in {"llama", "hf", "huggingface"}:
            return self.build_hf_tokenizer(model_id=model_id)
        if k.startswith("tiktoken"):
            if bool(append_bos) or bool(append_eos):
                raise ValueError("tiktoken tokenizer does not support append_bos/append_eos in prepare_fineweb_npy")
            enc = "cl100k_base"
            if ":" in k:
                enc = k.split(":", 1)[1].strip() or enc
            return self.build_tiktoken_tokenizer(encoding=enc)
        raise ValueError(f"Unsupported tokenizer {kind!r}. Use 'llama' or 'tiktoken[:encoding]'.")

    def build_hf_tokenizer(self, *, model_id: str) -> TextTokenizer:
        if not str(model_id).strip():
            raise ValueError("model_id must be non-empty for tokenizer=llama")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(model_id), use_fast=True, trust_remote_code=False)

        class _Tok:
            def encode(self, text: str) -> list[int]:
                return list(tok.encode(str(text), add_special_tokens=False))

            @property
            def bos_token_id(self) -> int | None:
                v = getattr(tok, "bos_token_id", None)
                return int(v) if v is not None else None

            @property
            def eos_token_id(self) -> int | None:
                v = getattr(tok, "eos_token_id", None)
                return int(v) if v is not None else None

        return _Tok()

    def build_tiktoken_tokenizer(self, *, encoding: str) -> TextTokenizer:
        import tiktoken

        enc = tiktoken.get_encoding(str(encoding))

        class _Tok:
            def encode(self, text: str) -> list[int]:
                return list(enc.encode(str(text)))

            @property
            def bos_token_id(self) -> int | None:
                return None

            @property
            def eos_token_id(self) -> int | None:
                return None

        return _Tok()

    def load_stream(self, *, dataset: str, subset: str | None, split: str):
        if not str(dataset).strip():
            raise ValueError("dataset must be non-empty")
        if not str(split).strip():
            raise ValueError("split must be non-empty")
        return load_dataset(path=str(dataset), name=str(subset) if subset else None, split=str(split), streaming=True)

    def open_memmap(self, *, path: Path, n_tokens: int) -> np.memmap:
        if int(n_tokens) < 2:
            raise ValueError("n_tokens must be >= 2")
        return np.lib.format.open_memmap(str(path), mode="w+", dtype=np.int32, shape=(int(n_tokens),))

    def write_tokens(
        self,
        *,
        mm: np.memmap,
        ds: Any,
        tokenizer: TextTokenizer,
        text_field: str,
        max_chars: int,
        append_bos: bool,
        append_eos: bool,
    ) -> dict[str, Any]:
        if not str(text_field).strip():
            raise ValueError("text_field must be non-empty")
        if int(max_chars) < 1:
            raise ValueError("max_chars must be >= 1")
        if bool(append_bos) and tokenizer.bos_token_id is None:
            raise ValueError("append_bos=true but tokenizer has no bos_token_id")
        if bool(append_eos) and tokenizer.eos_token_id is None:
            raise ValueError("append_eos=true but tokenizer has no eos_token_id")

        n_total = int(mm.shape[0])
        pos = 0
        t0 = time.time()
        pbar = tqdm(total=n_total, desc="prepare_fineweb_npy(tokens)", unit="tok")
        for ex in ds:
            if pos >= n_total:
                break
            if not isinstance(ex, dict):
                raise TypeError(f"Dataset example must be a dict, got {type(ex).__name__}")
            txt = ex.get(text_field)
            if not isinstance(txt, str):
                continue
            txt = txt[: int(max_chars)]
            ids = tokenizer.encode(txt)
            if bool(append_bos):
                ids = [int(tokenizer.bos_token_id), *ids]  # type: ignore[arg-type]
            if bool(append_eos):
                ids = [*ids, int(tokenizer.eos_token_id)]  # type: ignore[arg-type]
            if not ids:
                continue
            if any((int(x) < 0) for x in ids):
                raise ValueError("Tokenizer returned negative token id")
            take = min(len(ids), n_total - pos)
            mm[pos : pos + take] = np.asarray(ids[:take], dtype=np.int32)
            pos += int(take)
            pbar.update(int(take))
        pbar.close()
        if pos != n_total:
            raise RuntimeError(
                f"Dataset stream ended before reaching token budget: wrote={pos}, budget={n_total}. "
                "Increase data or lower tokens budget."
            )
        mm.flush()
        return {"tokens": n_total, "text_field": str(text_field), "max_chars": int(max_chars), "seconds": float(time.time() - t0)}

    def write_meta(self, *, out_path: Path, meta: dict[str, Any]) -> None:
        meta_path = out_path.with_suffix(out_path.suffix + ".meta")
        meta_path.write_text(str(meta), encoding="utf-8")


# Compatibility alias expected by trainers/datasets.
prepare_fineweb_npy = FinewebNpyPreparer().run

