"""Winogrande accuracy task

Loads and evaluates the Winogrande dataset, which tests commonsense reasoning
by asking models to resolve pronoun references in sentences.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from datasets import load_dataset

from benchmark.accuracy.tasks.base import BenchmarkAccuracyTask
from benchmark.accuracy.utils import DictCoercion, TextNormalization


class BenchmarkAccuracyTaskWinogrande(BenchmarkAccuracyTask):
    """Winogrande accuracy task"""

    def iter_examples(self, *, split: str) -> Iterator[tuple[str, list[str], str, str]]:
        """Iterate over Winogrande examples."""
        ds_wg: Any = load_dataset("winogrande", "winogrande_xl", split=str(split))
        for ex in ds_wg:
            exd = DictCoercion.as_dict(ex)
            sent = str(exd.get("sentence", ""))
            if "_" not in sent:
                continue
            pre, post = sent.split("_", 1)
            prompt = TextNormalization.norm_ws(pre).rstrip()
            if prompt and not prompt.endswith((" ", "\n")):
                prompt += " "
            opt1 = TextNormalization.norm_ws(exd.get("option1", ""))
            opt2 = TextNormalization.norm_ws(exd.get("option2", ""))
            suffix = TextNormalization.norm_ws(post)
            choices = [f"{opt1}{suffix}", f"{opt2}{suffix}"]
            ans = str(exd.get("answer", "")).strip()
            if ans == "1":
                gold = choices[0]
            elif ans == "2":
                gold = choices[1]
            else:
                continue
            yield prompt, choices, gold, str(split)
