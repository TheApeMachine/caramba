"""PiQA accuracy task

Loads and evaluates the Physical Interaction QA dataset, which tests physical
commonsense reasoning by asking models to choose the correct solution to
physical reasoning problems.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from datasets import load_dataset

from benchmark.accuracy.tasks.base import BenchmarkAccuracyTask
from benchmark.accuracy.utils import DictCoercion, TextNormalization


class BenchmarkAccuracyTaskPiqa(BenchmarkAccuracyTask):
    """PiQA accuracy task"""

    def iter_examples(self, *, split: str) -> Iterator[tuple[str, list[str], str, str]]:
        """Iterate over PiQA examples."""
        ds_piqa: Any = load_dataset("piqa", split=str(split))
        for ex in ds_piqa:
            exd = DictCoercion.as_dict(ex)
            goal = TextNormalization.norm_ws(exd.get("goal", ""))
            prompt = f"Goal: {goal}\nSolution:"
            sol1 = str(exd.get("sol1", ""))
            sol2 = str(exd.get("sol2", ""))
            choices = [" " + TextNormalization.norm_ws(sol1), " " + TextNormalization.norm_ws(sol2)]
            label = int(exd.get("label", -1))
            if label not in (0, 1):
                continue
            yield prompt, choices, choices[label], str(split)
