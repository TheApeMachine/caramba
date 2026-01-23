"""Base class for accuracy tasks

A benchmark accuracy task encapsulates dataset loading, example iteration,
and evaluation logic for a single accuracy benchmark task.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import nn

from collector.measurement.accuracy.task import TaskAccuracy, AccuracySample
from eval.logprob.scorer import LogprobScorer
from benchmark.accuracy.utils import TextNormalization
from benchmark.utils import LivePlotter
from console import logger


class BenchmarkAccuracyTask(ABC):
    """Base class for accuracy tasks

    Each task encapsulates its own dataset loading and example iteration logic.
    The base class provides the shared evaluation loop, logging, and result formatting.
    """

    def __init__(
        self,
        *,
        name: str,
        scorer: LogprobScorer,
        config: object,
        output_dir: Path | None,
        device: torch.device,
    ) -> None:
        self.name = name
        self.scorer = scorer
        self.config = config
        self.output_dir = output_dir
        self.device = device
        self._live_plotter: LivePlotter | None = None
        self._live_series: str | None = None

    def set_live_plotter(self, plotter: LivePlotter | None, *, series: str | None = None) -> None:
        self._live_plotter = plotter
        self._live_series = str(series) if series is not None else None

    @abstractmethod
    def iter_examples(self, *, split: str) -> Iterator[tuple[str, list[str], str, str]]:
        """Iterate over examples for this task.

        Yields (prompt, choices, gold_answer, split_name) tuples.
        """
        raise NotImplementedError

    def run(self, *, model: nn.Module, model_name: str) -> TaskAccuracy:
        """Run the task evaluation loop and return results."""
        limit = int(getattr(self.config, "limit", None) or 0) or None
        nshot = int(getattr(self.config, "num_fewshot", 0) or 0)
        nprint = int(getattr(self.config, "print_examples", 0) or 0)
        only_bad = bool(getattr(self.config, "print_only_incorrect", True))
        max_chars = int(getattr(self.config, "print_max_chars", 240) or 240)
        stream_live = bool(getattr(self.config, "stream_live", False))
        stream_every = int(getattr(self.config, "stream_every", 1) or 1)
        log_file = getattr(self.config, "log_file", None)

        # Build few-shot prefix if needed
        fewshot_prefix = ""
        if nshot > 0:
            try:
                k = 0
                for p, _choices, gold, _ in self.iter_examples(split="train"):
                    fewshot_prefix += f"{p}{gold}\n"
                    k += 1
                    if k >= nshot:
                        break
            except Exception:
                # If a task doesn't have a train split, keep 0-shot.
                fewshot_prefix = ""

        correct = 0
        total = 0
        split_name = "validation"
        shown = 0
        shown = 0
        sample_rows: list[list[str]] = []
        all_samples: list[AccuracySample] = []
        log_lines: list[str] = []

        def _tr(s: str, mx: int = max_chars) -> str:
            ss = TextNormalization.norm_ws(str(s))
            return ss if len(ss) <= mx else ss[: max(0, mx - 1)] + "…"

        # Print header showing which model is being evaluated
        if stream_live:
            logger.console.print()
            logger.console.print(
                f"[highlight]━━━ {self.name} • {model_name} ━━━[/highlight]"
            )

        # Log file header
        if log_file:
            log_lines.append(f"{'=' * 80}")
            log_lines.append(f"TASK: {self.name} | MODEL: {model_name}")
            log_lines.append(f"{'=' * 80}")
            log_lines.append("")

        # Track timing for speed comparison
        start_time = time.perf_counter()

        with torch.no_grad():
            for prompt, choices, gold, split in self.iter_examples(split="validation"):
                split_name = str(split)
                full_prompt = f"{fewshot_prefix}{prompt}"
                pred, scored = self.scorer.pick_choice_with_scores(
                    prompt=full_prompt, choices=list(choices)
                )
                total += 1
                ok = str(pred) == str(gold)
                if ok:
                    correct += 1

                # Live streaming: print each example as it's evaluated
                if stream_live and (total % stream_every == 0):
                    acc_so_far = (correct / total) * 100.0 if total > 0 else 0.0
                    status = "[success]✓[/success]" if ok else "[error]✗[/error]"
                    logger.console.print(
                        f"  {status} [{total:>5}] acc={acc_so_far:5.1f}% │ "
                        f"[muted]prompt:[/muted] {_tr(prompt, 50)} │ "
                        f"[muted]pred:[/muted] {_tr(pred, 30)} │ "
                        f"[muted]gold:[/muted] {_tr(gold, 30)}"
                    )
                    if self._live_plotter is not None:
                        try:
                            series = self._live_series or f"{model_name}:{self.name}"
                            self._live_plotter.log(**{series: float(acc_so_far)})
                        except Exception:
                            pass

                # Log file: write full untruncated details
                if log_file:
                    mark = "✓" if ok else "✗"
                    log_lines.append(f"[{total}] {mark}")
                    log_lines.append("-" * 40)
                    log_lines.append("PROMPT:")
                    log_lines.append(prompt)
                    log_lines.append("")
                    log_lines.append("CHOICES:")
                    for j, c in enumerate(choices):
                        log_lines.append(f"  [{j}] {c}")
                    log_lines.append("")
                    log_lines.append(f"GOLD: {gold}")
                    log_lines.append(f"PRED: {pred}")
                    log_lines.append("")

                # Collect samples for end-of-task table
                if nprint > 0 and shown < nprint:
                    if (not only_bad) or (not ok):
                        best_score = scored[0][1] if scored else float("nan")
                        gold_score = next(
                            (s for c, s in scored if str(c) == str(gold)), float("nan")
                        )
                        sample_rows.append(
                            [
                                "✓" if ok else "✗",
                                _tr(prompt),
                                _tr(gold),
                                _tr(pred),
                                f"{(best_score - gold_score):.2f}"
                                if (best_score == best_score and gold_score == gold_score)
                                else "",
                            ]
                        )
                        shown += 1

                # Collect for return object (capture everything)
                all_samples.append(AccuracySample(
                    prompt=prompt,
                    gold=str(gold),
                    pred=str(pred),
                    ok=ok
                ))

                # Always collect all samples for artifacts if we are logging
                # If nprint was limited, we might miss some in sample_rows if we relied only on that
                # But sample_rows seems to be controlled by nprint.
                # Let's decouple comprehensive logging from console printing.
                # We'll just capture EVERYTHING into sample_rows if we want full logs,
                # but `sample_rows` is used for the console table which might get too big.
                # Let's add a separate list for all samples.
                if limit is not None and total >= int(limit):
                    break

        acc = float(correct) / float(total) if total > 0 else 0.0
        elapsed = time.perf_counter() - start_time
        examples_per_sec = total / elapsed if elapsed > 0 else 0.0

        # Final summary for live streaming
        if stream_live:
            logger.console.print()
            logger.console.print(
                f"  [highlight]━━━ {self.name} complete:[/highlight] "
                f"[metric]{correct}[/metric]/[metric]{total}[/metric] = "
                f"[success]{acc * 100.0:.2f}%[/success] accuracy "
                f"[muted]({elapsed:.1f}s, {examples_per_sec:.1f} ex/s)[/muted]"
            )
            logger.console.print()

        # Write log file (append mode to accumulate across models)
        if log_file and log_lines:
            log_lines.append(f"SUMMARY: {correct}/{total} = {acc * 100:.2f}%")
            log_lines.append(f"TIME: {elapsed:.2f}s ({examples_per_sec:.2f} examples/sec)")
            log_lines.append("")
            log_lines.append("")

            log_path = Path(log_file)
            if self.output_dir and not log_path.is_absolute():
                log_path = self.output_dir / log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write("\n".join(log_lines) + "\n")
            logger.info(f"Accuracy log appended to: {log_path}")

        logger.metric(str(model_name), acc * 100.0, "% acc")
        logger.metric(f"{model_name}_time", elapsed, "s")
        if sample_rows:
            logger.table(
                title=f"Accuracy examples • {self.name} • {model_name}",
                columns=["ok", "prompt", "gold", "pred", "Δ(best-gold)"],
                rows=sample_rows,
            )
        return TaskAccuracy(
            task=str(self.name),
            split=str(split_name),
            accuracy=float(acc),
            correct=int(correct),
            total=int(total),
            samples=all_samples,
            elapsed_seconds=float(elapsed),
        )
