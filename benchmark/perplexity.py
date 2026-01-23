"""Perplexity benchmark: measuring language modeling quality.

Perplexity is exp(average cross-entropy loss) over tokens. Lower is better.
For upcycling, we want the student's perplexity to be close to the teacher's,
proving we've preserved model quality while changing the architecture.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from console import logger
from config.benchmark import PerplexityBenchmarkConfig
from data.npy import NpyDataset
from benchmark.utils import LivePlotter, get_model_vocab_size
from runtime.tensordict_utils import TensorDictBase, collate_tensordict


@dataclass
class PerplexityResult:
    """Results from a perplexity benchmark run."""

    model_name: str
    perplexity: float
    loss: float
    num_tokens: int
    num_batches: int
    # Full audit trail of the batch-level numbers used to compute loss/ppl.
    # Each entry corresponds to one processed batch (sum reduction over tokens).
    batch_loss_sums: list[float]
    batch_token_counts: list[int]


class PerplexityBenchmark:
    """Measures perplexity on a token dataset.

    Computes cross-entropy loss over the dataset and converts to perplexity.
    The dataset is cached so multiple models can be evaluated without
    reloading the data.
    """

    def __init__(self, config: PerplexityBenchmarkConfig, device: torch.device) -> None:
        """Set up the benchmark with config and target device."""
        self.config = config
        self.device = device
        self._dataset: NpyDataset | None = None
        self._loader: DataLoader[TensorDictBase] | None = None

    def _get_loader(self) -> DataLoader[TensorDictBase]:
        """Lazily initialize and cache the dataset and dataloader."""
        if self._dataset is None:
            self._dataset = NpyDataset(self.config.dataset, block_size=self.config.block_size)

        logger.info(f"PerplexityBenchmark: dataset path: {self.config.dataset}")
        logger.info(f"PerplexityBenchmark: dataset length: {len(self._dataset)}")

        if self._loader is None:
            self._loader = DataLoader(
                self._dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                # IMPORTANT: don't drop the tail batch; perplexity should cover the whole dataset.
                drop_last=False,
                collate_fn=collate_tensordict,
            )
        return self._loader

    def run(
        self,
        model: nn.Module,
        model_name: str,
        *,
        plotter: LivePlotter | None = None,
        plot_series: str | None = None,
    ) -> PerplexityResult:
        """Run the perplexity benchmark on a model.

        Iterates through the dataset, computing cross-entropy loss for each
        batch, then converts total loss to perplexity.
        """
        model.eval()
        loader = self._get_loader()

        model_vocab_size = get_model_vocab_size(model, default=32000)
        effective_vocab_size = (
            int(self.config.valid_vocab_size)
            if self.config.valid_vocab_size is not None
            else int(model_vocab_size)
        )
        if int(effective_vocab_size) > int(model_vocab_size):
            raise ValueError(
                "PerplexityBenchmark: valid_vocab_size exceeds model vocab_size "
                f"(valid_vocab_size={effective_vocab_size}, model_vocab_size={model_vocab_size})."
            )

        # Optional ignore index support. If your config doesn't define it, this stays None.
        ignore_index = getattr(self.config, "ignore_index", None)
        if ignore_index is not None:
            ignore_index = int(ignore_index)

        # Optional AMP (mixed precision) for eval. If your config doesn't define it, defaults to False.
        use_amp = bool(getattr(self.config, "use_amp", False))
        # NOTE: torch.autocast("mps", ...) only supports fp16/bf16, and even when
        # autocast is disabled it may still validate dtype and emit warnings.
        # So we only enter the autocast context when actually enabled.
        amp_enabled = bool(use_amp and (self.device.type in ("cuda", "mps")))
        amp_dtype: torch.dtype = torch.float16
        cfg_amp_dtype = getattr(self.config, "amp_dtype", None)
        if isinstance(cfg_amp_dtype, str):
            s = cfg_amp_dtype.strip().lower()
            if s in ("bf16", "bfloat16", "torch.bfloat16"):
                amp_dtype = torch.bfloat16
            elif s in ("fp16", "float16", "torch.float16"):
                amp_dtype = torch.float16
        elif isinstance(cfg_amp_dtype, torch.dtype):
            amp_dtype = cfg_amp_dtype

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        batch_loss_sums: list[float] = []
        batch_token_counts: list[int] = []

        # Token ID sanity checking can be expensive if it syncs GPU every batch.
        # We check the first batch and then periodically.
        check_every = int(getattr(self.config, "token_id_check_every_n_batches", 100))
        if check_every <= 0:
            check_every = 100

        series = str(plot_series or model_name or "loss")

        # inference_mode is slightly faster/more memory-friendly than no_grad for eval-only.
        with torch.inference_mode():
            for batch in loader:
                x = cast(Tensor, batch["input_ids"]).to(self.device)
                y = cast(Tensor, batch["target_ids"]).to(self.device)

                # Fail fast if token IDs exceed the effective vocab (tokenizer mismatch => meaningless ppl).
                # Do this on the first batch and then periodically to reduce sync overhead.
                if num_batches == 0 or (num_batches % check_every == 0):
                    try:
                        mx = int(torch.maximum(x.max(), y.max()).item())
                    except Exception as e:
                        logger.error(f"Failed to compute max token ID: {e}")
                        raise
                    if mx >= int(effective_vocab_size):
                        raise ValueError(
                            f"Dataset token IDs exceed effective vocab: max_id={mx}, "
                            f"valid_vocab_size={effective_vocab_size} (model_vocab_size={model_vocab_size}). "
                            "This usually means the dataset was tokenized with a different tokenizer than the "
                            "one implied by valid_vocab_size (or the model)."
                        )

                # Forward pass (optionally autocast for speed on CUDA/MPS).
                if amp_enabled:
                    with torch.autocast(device_type=self.device.type, enabled=True, dtype=amp_dtype):
                        logits = model(x)
                else:
                    logits = model(x)

                if isinstance(logits, tuple):
                    logits = logits[0]
                elif hasattr(logits, "logits"):
                    logits = logits.logits

                logits = cast(Tensor, logits)

                # Ensure logits vocab covers our effective vocab.
                if int(logits.size(-1)) < int(effective_vocab_size):
                    raise ValueError(
                        "Model returned logits with vocab smaller than effective vocab "
                        f"(logits_vocab={int(logits.size(-1))}, effective_vocab_size={effective_vocab_size})."
                    )

                # If model vocab is padded, slice logits so loss is computed over the real token space.
                if int(logits.size(-1)) > int(effective_vocab_size):
                    logits = logits[..., : int(effective_vocab_size)]

                # Flatten safely (reshape handles non-contiguous tensors).
                vocab = int(logits.size(-1))
                logits_2d = logits.reshape(-1, vocab)
                y_1d = y.reshape(-1)

                # Sum reduction so we can correctly compute dataset-level average loss.
                loss_sum = F.cross_entropy(
                    logits_2d,
                    y_1d,
                    reduction="sum",
                    ignore_index=ignore_index if ignore_index is not None else -100,  # set below if None
                )

                # If ignore_index wasn't actually intended, don't silently ignore -100 labels.
                # We only pass ignore_index when configured; otherwise use default behavior.
                if ignore_index is None:
                    loss_sum = F.cross_entropy(logits_2d, y_1d, reduction="sum")

                # Compute the token count used for averaging.
                if ignore_index is None:
                    batch_tokens = int(y_1d.numel())
                else:
                    batch_tokens = int((y_1d != ignore_index).sum().item())

                if plotter is not None:
                    # Raw batch loss (sum over tokens); useful as a stability/variance signal.
                    plotter.log(**{series: float(loss_sum)})

                total_loss += float(loss_sum)
                total_tokens += batch_tokens
                num_batches += 1
                batch_loss_sums.append(float(loss_sum))
                batch_token_counts.append(int(batch_tokens))

                if self.config.num_batches and num_batches >= self.config.num_batches:
                    break

        if total_tokens <= 0:
            raise ValueError(
                f"PerplexityBenchmark: no tokens processed (num_batches={num_batches}). "
                "This indicates a dataset/loader configuration error; refusing to emit inf/0."
            )

        avg_loss = total_loss / total_tokens
        if not math.isfinite(float(avg_loss)) or float(avg_loss) < 0.0:
            raise ValueError(f"PerplexityBenchmark: avg_loss must be finite and >= 0, got {avg_loss!r}")

        # Only guard actual overflow, not "too big to be meaningful".
        try:
            perplexity = math.exp(avg_loss) if avg_loss < 700.0 else float("inf")
        except OverflowError:
            perplexity = float("inf")

        if not math.isfinite(float(perplexity)) or float(perplexity) <= 0.0:
            raise ValueError(f"PerplexityBenchmark: perplexity must be finite and > 0, got {perplexity!r}")

        return PerplexityResult(
            model_name=model_name,
            perplexity=float(perplexity),
            loss=float(avg_loss),
            num_tokens=int(total_tokens),
            num_batches=int(num_batches),
            batch_loss_sums=batch_loss_sums,
            batch_token_counts=batch_token_counts,
        )
