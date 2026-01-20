"""HellaSwag accuracy task

Loads and evaluates the HellaSwag dataset, which tests commonsense reasoning
by asking models to choose the most plausible continuation of a given context.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

from datasets import load_dataset

from benchmark.accuracy.tasks.base import BenchmarkAccuracyTask
from benchmark.accuracy.utils import DictCoercion, TextNormalization


class BenchmarkAccuracyTaskHellaswag(BenchmarkAccuracyTask):
    """HellaSwag accuracy task"""

    def iter_examples(self, *, split: str) -> Iterator[tuple[str, list[str], str, str]]:
        """Iterate over HellaSwag examples."""
        ds_hs: Any = load_dataset("hellaswag", split=str(split))
        for ex in ds_hs:
            exd = DictCoercion.as_dict(ex)
            ctx_a = TextNormalization.norm_ws(exd.get("ctx_a", ""))
            ctx_b = TextNormalization.norm_ws(exd.get("ctx_b", ""))
            prompt = (ctx_a + " " + ctx_b).strip()
            if prompt and not prompt.endswith((" ", "\n")):
                prompt += " "
            endings = cast(list[str], exd.get("endings", []))
            endings = [str(e) for e in endings]
            label = int(exd.get("label", -1))
            if label < 0 or label >= len(endings):
                continue
            yield prompt, endings, str(endings[label]), str(split)
