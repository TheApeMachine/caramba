"""Downstream-task accuracy benchmark (HF datasets).

This benchmark evaluates standard multiple-choice / classification style tasks
by scoring answer options via next-token log-probabilities (no instruction tuning
assumed). It pulls full datasets using HuggingFace `datasets`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from config.benchmark import AccuracyBenchmarkConfig
from console import logger
from data.tokenizers.builder import TokenizerBuilder

from collector.measurement.accuracy.result import AccuracyResult

from eval.logprob.scorer import LogprobScorer
from eval.logprob.completion.full_sequence import LogprobCompletionFullSequence
from eval.logprob.completion.windowed import LogprobCompletionWindowed
from benchmark.utils import get_model_vocab_size

from benchmark.accuracy.tasks.builder import BenchmarkAccuracyTaskBuilder


class BenchmarkAccuracy:
    """Evaluate standard downstream datasets using logprob-scored choices."""

    def __init__(self, config: AccuracyBenchmarkConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.tokenizer = TokenizerBuilder().build(self.config.tokenizer)
        self._output_dir: Path | None = None

    def run(
        self,
        model: nn.Module,
        model_name: str,
        output_dir: Path | None = None,
    ) -> AccuracyResult:
        """Run accuracy benchmark across all configured tasks."""
        model.eval()
        self._output_dir = output_dir
        out = AccuracyResult(model_name=str(model_name))

        # Determine tokenizer vocab size for fair scoring when model vocab is padded.
        tok_vocab: int | None = None
        for attr in ("n_vocab", "vocab_size"):
            if hasattr(self.tokenizer, attr):
                try:
                    tok_vocab = int(getattr(self.tokenizer, attr))
                    break
                except Exception:
                    tok_vocab = None
        if tok_vocab is None and hasattr(self.tokenizer, "_enc"):
            try:
                tok_vocab = int(getattr(getattr(self.tokenizer, "_enc"), "n_vocab"))
            except Exception:
                tok_vocab = None

        if tok_vocab is not None:
            mv = int(get_model_vocab_size(model, default=0))
            if mv > 0 and mv < int(tok_vocab):
                raise ValueError(
                    "Accuracy benchmark tokenizer/model vocab mismatch: "
                    f"tokenizer_vocab={int(tok_vocab)} > model_vocab={mv}. "
                    "This likely indicates evaluating with the wrong tokenizer."
                )

        # Build completion scorer (windowed or full-sequence)
        ctxw = int(self.config.context_window) if self.config.context_window is not None else None
        completion = (
            LogprobCompletionWindowed(
                model=model,
                device=self.device,
                context_window=int(ctxw),
                valid_vocab_size=tok_vocab,
            )
            if ctxw is not None
            else LogprobCompletionFullSequence(model=model, device=self.device, valid_vocab_size=tok_vocab)
        )
        scorer = LogprobScorer(tokenizer=self.tokenizer, completion=completion)

        # Build task builder
        builder = BenchmarkAccuracyTaskBuilder(
            scorer=scorer,
            config=self.config,
            output_dir=self._output_dir,
            device=self.device,
        )

        # Run each configured task
        for task_name in list(self.config.tasks):
            t = str(task_name).strip().lower()
            if not t:
                continue
            logger.subheader(f"accuracy:{t}")
            task = builder.build(t)
            result = task.run(model=model, model_name=model_name)
            out.tasks.append(result)

        return out
