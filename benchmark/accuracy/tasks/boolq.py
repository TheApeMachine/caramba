"""BoolQ accuracy task

Loads and evaluates the BoolQ dataset, which tests reading comprehension by
asking yes/no questions about passages.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from datasets import load_dataset

from caramba.benchmark.accuracy.tasks.base import BenchmarkAccuracyTask
from caramba.benchmark.accuracy.utils import DictCoercion, TextNormalization


class BenchmarkAccuracyTaskBoolq(BenchmarkAccuracyTask):
    """BoolQ accuracy task"""

    def iter_examples(self, *, split: str) -> Iterator[tuple[str, list[str], str, str]]:
        """Iterate over BoolQ examples."""
        ds_bq: Any = load_dataset("boolq", split=str(split))
        for ex in ds_bq:
            exd = DictCoercion.as_dict(ex)
            passage = TextNormalization.norm_ws(exd.get("passage", ""))
            question = TextNormalization.norm_ws(exd.get("question", ""))
            prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
            choices = [" yes", " no"]
            ans = exd.get("answer", None)
            if not isinstance(ans, (bool, int)):
                continue
            gold = choices[0] if bool(ans) else choices[1]
            yield prompt, choices, gold, str(split)
