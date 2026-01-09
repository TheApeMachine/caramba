"""Task builder for building accuracy benchmark tasks from config."""
from __future__ import annotations

from pathlib import Path

import torch

from caramba.eval.logprob.scorer import LogprobScorer
from caramba.data.tokenizers.base import Tokenizer

from caramba.benchmark.accuracy.tasks.base import BenchmarkAccuracyTask
from caramba.benchmark.accuracy.tasks.hellaswag import BenchmarkAccuracyTaskHellaswag
from caramba.benchmark.accuracy.tasks.piqa import BenchmarkAccuracyTaskPiqa
from caramba.benchmark.accuracy.tasks.winogrande import BenchmarkAccuracyTaskWinogrande
from caramba.benchmark.accuracy.tasks.arc_easy import BenchmarkAccuracyTaskArcEasy
from caramba.benchmark.accuracy.tasks.arc_challenge import BenchmarkAccuracyTaskArcChallenge
from caramba.benchmark.accuracy.tasks.boolq import BenchmarkAccuracyTaskBoolq


class BenchmarkAccuracyTaskBuilder:
    """Builds task instances from task name strings."""

    def __init__(
        self,
        *,
        scorer: LogprobScorer,
        config: object,
        output_dir: Path | None,
        device: torch.device,
    ) -> None:
        self.scorer = scorer
        self.config = config
        self.output_dir = output_dir
        self.device = device

    def build(self, task_name: str) -> BenchmarkAccuracyTask:
        """Build a task instance from its name."""
        name = str(task_name).strip().lower()
        if name == "hellaswag":
            return BenchmarkAccuracyTaskHellaswag(
                name=name,
                scorer=self.scorer,
                config=self.config,
                output_dir=self.output_dir,
                device=self.device,
            )
        elif name == "piqa":
            return BenchmarkAccuracyTaskPiqa(
                name=name,
                scorer=self.scorer,
                config=self.config,
                output_dir=self.output_dir,
                device=self.device,
            )
        elif name == "winogrande":
            return BenchmarkAccuracyTaskWinogrande(
                name=name,
                scorer=self.scorer,
                config=self.config,
                output_dir=self.output_dir,
                device=self.device,
            )
        elif name == "arc_easy":
            return BenchmarkAccuracyTaskArcEasy(
                name=name,
                scorer=self.scorer,
                config=self.config,
                output_dir=self.output_dir,
                device=self.device,
            )
        elif name == "arc_challenge":
            return BenchmarkAccuracyTaskArcChallenge(
                name=name,
                scorer=self.scorer,
                config=self.config,
                output_dir=self.output_dir,
                device=self.device,
            )
        elif name == "boolq":
            return BenchmarkAccuracyTaskBoolq(
                name=name,
                scorer=self.scorer,
                config=self.config,
                output_dir=self.output_dir,
                device=self.device,
            )
        else:
            raise ValueError(
                f"Unsupported accuracy task {task_name!r}. "
                "Supported: hellaswag, piqa, winogrande, arc_easy, arc_challenge, boolq."
            )
