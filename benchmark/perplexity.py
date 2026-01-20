"""Perplexity benchmark: measuring language modeling quality.

Perplexity is exp(average cross-entropy loss) over tokens. Lower is better.
For upcycling, we want the student's perplexity to be close to the teacher's,
proving we've preserved model quality while changing the architecture.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from console import logger
from config.benchmark import PerplexityBenchmarkConfig
from data.npy import NpyDataset
from benchmark.utils import get_model_vocab_size
from runtime.tensordict_utils import TensorDictBase, collate_tensordict


@dataclass
class PerplexityResult:
    """Results from a perplexity benchmark run."""

    model_name: str
    perplexity: float
    loss: float
    num_tokens: int
    num_batches: int


class PerplexityBenchmark:
    """Measures perplexity on a token dataset.

    Computes cross-entropy loss over the dataset and converts to perplexity.
    The dataset is cached so multiple models can be evaluated without
    reloading the data.
    """

    def __init__(
        self, config: PerplexityBenchmarkConfig, device: torch.device
    ) -> None:
        """Set up the benchmark with config and target device."""
        self.config = config
        self.device = device
        self._dataset: NpyDataset | None = None
        self._loader: DataLoader[TensorDictBase] | None = None

    def _get_loader(self) -> DataLoader[TensorDictBase]:
        """Lazily initialize and cache the dataset and dataloader."""
        if self._dataset is None:
            self._dataset = NpyDataset(
                self.config.dataset, block_size=self.config.block_size
            )
        if self._loader is None:
            self._loader = DataLoader(
                self._dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=True,
                collate_fn=collate_tensordict,
            )
        return self._loader

    def run(self, model: nn.Module, model_name: str) -> PerplexityResult:
        """Run the perplexity benchmark on a model.

        Iterates through the dataset, computing cross-entropy loss for each
        batch, then converts total loss to perplexity.
        """
        model.eval()
        loader = self._get_loader()
        model_vocab_size = get_model_vocab_size(model, default=32000)
        effective_vocab_size = (
            int(self.config.valid_vocab_size)
            if getattr(self.config, "valid_vocab_size", None) is not None
            else int(model_vocab_size)
        )
        if int(effective_vocab_size) > int(model_vocab_size):
            raise ValueError(
                "PerplexityBenchmark: valid_vocab_size exceeds model vocab_size "
                f"(valid_vocab_size={effective_vocab_size}, model_vocab_size={model_vocab_size})."
            )

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        warned_max_failed = False

        with torch.no_grad():
            for batch in loader:
                x = cast(Tensor, batch["input_ids"]).to(self.device)
                y = cast(Tensor, batch["target_ids"]).to(self.device)

                # Fail fast if the dataset token IDs exceed the model vocab.
                # This frequently indicates a tokenizer mismatch (e.g. GPT-family tokens
                # evaluated with a Llama-family model), which makes perplexity meaningless.
                # OPTIMIZED: Stay on GPU until final .item() to avoid sync overhead.
                try:
                    # Compute max on GPU, only transfer final scalar to CPU
                    mx = int(max(x.max().item(), y.max().item()))
                except Exception as e:
                    # If max() fails for some reason, continue and let cross_entropy raise.
                    if not warned_max_failed:
                        warned_max_failed = True
                        logger.warning(
                            f"Max() failed in vocab guard ({type(e).__name__}: {e}); "
                            "continuing and letting cross_entropy raise"
                        )
                else:
                    if mx >= int(effective_vocab_size):
                        raise ValueError(
                            f"Dataset token IDs exceed effective vocab: max_id={mx}, "
                            f"valid_vocab_size={effective_vocab_size} (model_vocab_size={model_vocab_size}). "
                            "This usually means the dataset was tokenized with a different tokenizer than the "
                            "one implied by valid_vocab_size (or the model)."
                        )

                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                elif hasattr(logits, "logits"):
                    logits = logits.logits
                logits = cast(Tensor, logits)
                # If model vocab is padded (e.g. 50304 vs tokenizer 50257), slice logits
                # so loss is computed over the *real* token space for fair comparison.
                if int(logits.size(-1)) < int(effective_vocab_size):
                    raise ValueError(
                        "Model returned logits with vocab smaller than effective vocab "
                        f"(logits_vocab={int(logits.size(-1))}, effective_vocab_size={effective_vocab_size})."
                    )
                if int(logits.size(-1)) > int(effective_vocab_size):
                    logits = logits[..., : int(effective_vocab_size)]

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    reduction="sum",
                )

                batch_tokens = y.numel()
                total_loss += float(loss)
                total_tokens += batch_tokens
                num_batches += 1

                if self.config.num_batches and num_batches >= self.config.num_batches:
                    break

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")

        return PerplexityResult(
            model_name=model_name,
            perplexity=perplexity,
            loss=avg_loss,
            num_tokens=total_tokens,
            num_batches=num_batches,
        )
