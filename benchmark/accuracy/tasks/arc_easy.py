"""ARC-Easy accuracy task

Loads and evaluates the ARC-Easy dataset, which tests science question answering
with easier questions that require less domain knowledge.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

from datasets import load_dataset

from caramba.benchmark.accuracy.tasks.base import BenchmarkAccuracyTask
from caramba.benchmark.accuracy.utils import DictCoercion, TextNormalization


class BenchmarkAccuracyTaskArcEasy(BenchmarkAccuracyTask):
    """ARC-Easy accuracy task"""

    def iter_examples(self, *, split: str) -> Iterator[tuple[str, list[str], str, str]]:
        """Iterate over ARC-Easy examples."""
        ds_arc: Any = load_dataset("ai2_arc", "ARC-Easy", split=str(split))
        for ex in ds_arc:
            exd = DictCoercion.as_dict(ex)
            q = TextNormalization.norm_ws(exd.get("question", ""))
            prompt = f"Question: {q}\nAnswer:"
            ch = exd.get("choices", {})
            labels = cast(list[str], ch.get("label", [])) if isinstance(ch, dict) else []
            texts = cast(list[str], ch.get("text", [])) if isinstance(ch, dict) else []
            if not labels or not texts or len(labels) != len(texts):
                continue
            # Use choice texts as completions (prefixed with space).
            choices = [" " + TextNormalization.norm_ws(t) for t in texts]
            key = str(exd.get("answerKey", "")).strip()
            if not key:
                continue
            try:
                idx = [str(x).strip() for x in labels].index(key)
            except ValueError:
                continue
            gold = choices[int(idx)]
            yield prompt, choices, gold, str(split)
