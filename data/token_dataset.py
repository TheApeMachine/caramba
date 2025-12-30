"""Dataset components.

These are manifest-referenced datasets that can be built into torch Datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from torch.utils.data import Dataset

from data.auto import build_token_dataset
from runtime.tensordict_utils import TensorDictBase

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TokenDataset:
    """Token dataset component for next-token training.

    Config:
    - path: dataset file path (.npy/.tokens/.txt)
    - block_size: sequence length
    """

    path: str
    block_size: int
    # Optional manifest-driven dataset preparation.
    # Example:
    #   prepare:
    #     type: fineweb
    #     tokens: 100M
    #     tokenizer: llama
    #     model_id: meta-llama/Llama-3.2-1B
    #     dataset: HuggingFaceFW/fineweb-edu
    #     subset: null
    #     split: train
    #     text_field: text
    #     max_chars: 50000
    #     append_eos: true
    #     append_bos: false
    #     rebuild_on_failure: true
    prepare: dict[str, object] | None = None

    def build(self) -> Dataset[TensorDictBase]:
        p = Path(self.path)
        try:
            return build_token_dataset(path=p, block_size=int(self.block_size))
        except (FileNotFoundError, EOFError, ValueError) as e:
            _log.warning(
                "Failed to build/load token dataset; will try prepare if configured "
                "(path=%s, block_size=%s): %s",
                p,
                self.block_size,
                e,
                exc_info=True,
            )
            # If the dataset is missing or malformed, and a prepare config is present,
            # build it now and retry.
            if not isinstance(self.prepare, dict):
                raise

        cfg = self.prepare
        kind = str(cfg.get("type", "fineweb")).lower().strip()
        if kind not in {"fineweb", "fineweb_npy"}:
            raise ValueError(f"Unsupported data.config.prepare.type={kind!r}")

        # Import locally so training can run without dataset deps unless requested.
        from prepare_fineweb import prepare_fineweb_npy

        tok = str(cfg.get("tokenizer", "llama"))
        model_id = str(cfg.get("model_id", "meta-llama/Llama-3.2-1B"))
        ds_id = str(cfg.get("dataset", "HuggingFaceFW/fineweb"))
        subset = cfg.get("subset", None)
        subset_s = str(subset) if isinstance(subset, str) and subset else None
        split = str(cfg.get("split", "train"))
        text_field = str(cfg.get("text_field", "text"))
        try:
            max_chars = int(cfg.get("max_chars", 50_000))  # type: ignore[arg-type]
        except Exception:
            max_chars = 50_000
        append_eos = bool(cfg.get("append_eos", True))
        append_bos = bool(cfg.get("append_bos", False))
        token_budget = str(cfg.get("tokens", "100M"))
        rebuild_on_failure = bool(cfg.get("rebuild_on_failure", True))

        # First try to build without overwrite (respects existing files).
        prepare_fineweb_npy(
            tokens=token_budget,
            output=str(p),
            tokenizer=tok,
            model_id=model_id,
            dataset=ds_id,
            subset=subset_s,
            split=split,
            text_field=text_field,
            max_chars=max_chars,
            append_eos=append_eos,
            append_bos=append_bos,
            overwrite=False,
        )
        try:
            return build_token_dataset(path=p, block_size=int(self.block_size))
        except Exception as e:
            _log.warning(
                "Token dataset build failed after prepare (overwrite=False); "
                "will retry with overwrite=True if allowed "
                "(path=%s, block_size=%s, rebuild_on_failure=%s): %s",
                p,
                self.block_size,
                rebuild_on_failure,
                e,
                exc_info=True,
            )
            if not rebuild_on_failure:
                raise

        # Retry with overwrite to recover from mismatched tokenizer/config.
        prepare_fineweb_npy(
            tokens=token_budget,
            output=str(p),
            tokenizer=tok,
            model_id=model_id,
            dataset=ds_id,
            subset=subset_s,
            split=split,
            text_field=text_field,
            max_chars=max_chars,
            append_eos=append_eos,
            append_bos=append_bos,
            overwrite=True,
        )
        return build_token_dataset(path=p, block_size=int(self.block_size))

