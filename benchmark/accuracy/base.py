"""Downstream-task accuracy benchmark (HF datasets).

This benchmark evaluates standard multiple-choice / classification style tasks
by scoring answer options via next-token log-probabilities (no instruction tuning
assumed). It pulls full datasets using HuggingFace `datasets`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from caramba.config.benchmark import AccuracyBenchmarkConfig
from caramba.console import logger
from caramba.data.tokenizers.builder import TokenizerBuilder

from caramba.trainer.collector.measurement.accuracy.result import AccuracyResult

from caramba.eval.logprob.scorer import LogprobScorer
from caramba.eval.logprob.completion.full_sequence import LogprobCompletionFullSequence
from caramba.eval.logprob.completion.windowed import LogprobCompletionWindowed

from caramba.benchmark.accuracy.tasks.builder import BenchmarkAccuracyTaskBuilder


class BenchmarkAccuracy:
    """Evaluate standard downstream datasets using logprob-scored choices."""

    def __init__(self, config: AccuracyBenchmarkConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.tokenizer = TokenizerBuilder().build_tokenizer(self.config.tokenizer)
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

        # Build completion scorer (windowed or full-sequence)
        ctxw = int(self.config.context_window) if self.config.context_window is not None else None
        completion = (
            LogprobCompletionWindowed(model=model, device=self.device, context_window=int(ctxw))
            if ctxw is not None
            else LogprobCompletionFullSequence(model=model, device=self.device)
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
