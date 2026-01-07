"""Downstream-task accuracy benchmark (HF datasets).

This benchmark evaluates standard multiple-choice / classification style tasks
by scoring answer options via next-token log-probabilities (no instruction tuning
assumed). It pulls full datasets using HuggingFace `datasets`.

Supported tasks (task ids in config):
- hellaswag
- piqa
- winogrande
- arc_easy
- arc_challenge
- boolq
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, cast

import torch
import torch.nn.functional as F
from torch import nn

from caramba.config.benchmark import AccuracyBenchmarkConfig
from caramba.console import logger
from caramba.eval.tokenizer import Tokenizer, build_tokenizer


@dataclass
class TaskAccuracy:
    task: str
    split: str
    accuracy: float
    correct: int
    total: int


@dataclass
class AccuracyResult:
    model_name: str
    tasks: list[TaskAccuracy] = field(default_factory=list)

    @property
    def micro_accuracy(self) -> float:
        """Overall accuracy weighted by number of examples."""
        tot = sum(int(t.total) for t in self.tasks)
        if tot <= 0:
            return 0.0
        cor = sum(int(t.correct) for t in self.tasks)
        return float(cor) / float(tot)


def _score_completion_logprob(
    *,
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    completion: str,
    device: torch.device,
    context_window: int | None,
) -> float:
    """Score a completion by summing token log-probabilities under the model."""
    prompt_ids = tokenizer.encode(prompt)
    completion_ids = tokenizer.encode(completion)
    if not prompt_ids or not completion_ids:
        # If encoding collapses (rare), make it very unlikely.
        return float("-inf")

    if context_window is not None:
        # Windowed scoring: score each completion token with a sliding context.
        seq = prompt_ids + completion_ids
        total = 0.0
        start_k = len(prompt_ids)
        for k in range(start_k, len(seq)):
            ctx = seq[max(0, k - int(context_window)) : k]
            if not ctx:
                continue
            x = torch.tensor([ctx], device=device, dtype=torch.long)
            logits = model(x)
            lp = F.log_softmax(logits[0, -1, :], dim=-1)
            total += float(lp[int(seq[k])])
        return float(total)

    seq = prompt_ids + completion_ids
    x = torch.tensor([seq], device=device, dtype=torch.long)
    logits = model(x)
    if logits.ndim != 3:
        raise ValueError(f"Expected logits (B,T,V), got {tuple(logits.shape)}")
    if int(logits.shape[1]) != len(seq):
        raise ValueError("Unexpected logits length mismatch")

    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = x[:, 1:]
    start = len(prompt_ids) - 1
    end = start + len(completion_ids)
    tok_logp = logp[0, start:end, :].gather(
        dim=-1,
        index=target[0, start:end].unsqueeze(-1),
    )
    return float(tok_logp.sum())


def _pick_choice_by_logprob(
    *,
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    choices: list[str],
    device: torch.device,
    context_window: int | None,
) -> str:
    if not choices:
        raise ValueError("choices must be non-empty")
    best: tuple[float, str] | None = None
    for c in choices:
        s = _score_completion_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            completion=str(c),
            device=device,
            context_window=context_window,
        )
        item = (float(s), str(c))
        best = item if best is None or item[0] > best[0] else best
    assert best is not None
    return best[1]

def _pick_choice_with_scores(
    *,
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    choices: list[str],
    device: torch.device,
    context_window: int | None,
) -> tuple[str, list[tuple[str, float]]]:
    """Pick best choice and also return per-choice scores."""
    scored: list[tuple[str, float]] = []
    for c in choices:
        s = _score_completion_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            completion=str(c),
            device=device,
            context_window=context_window,
        )
        scored.append((str(c), float(s)))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[0][0], scored


def _norm_ws(s: str) -> str:
    return " ".join(str(s).replace("\r\n", "\n").replace("\r", "\n").split())

def _as_dict(x: Any) -> dict[str, Any]:
    """Best-effort conversion from dataset row to dict."""
    return x if isinstance(x, dict) else {}


class AccuracyBenchmark:
    """Evaluate standard downstream datasets using logprob-scored choices."""

    def __init__(self, config: AccuracyBenchmarkConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.tokenizer = build_tokenizer(self.config.tokenizer)
        self._output_dir: Path | None = None

    def run(
        self,
        model: nn.Module,
        model_name: str,
        output_dir: Path | None = None,
    ) -> AccuracyResult:
        model.eval()
        self._output_dir = output_dir
        out = AccuracyResult(model_name=str(model_name))

        for task in list(self.config.tasks):
            t = str(task).strip().lower()
            if not t:
                continue
            logger.subheader(f"accuracy:{t}")
            out.tasks.append(self._run_task(model=model, model_name=model_name, task=t))

        return out

    def _run_task(self, *, model: nn.Module, model_name: str, task: str) -> TaskAccuracy:
        # Import lazily: datasets is heavy and only needed for accuracy.
        from datasets import load_dataset  # type: ignore

        limit = int(self.config.limit) if self.config.limit is not None else None
        ctxw = int(self.config.context_window) if self.config.context_window is not None else None
        nshot = int(self.config.num_fewshot) if int(self.config.num_fewshot) > 0 else 0
        nprint = int(getattr(self.config, "print_examples", 0))
        only_bad = bool(getattr(self.config, "print_only_incorrect", True))
        max_chars = int(getattr(self.config, "print_max_chars", 240))
        stream_live = bool(getattr(self.config, "stream_live", False))
        stream_every = int(getattr(self.config, "stream_every", 1))

        # Task adapters return (prompt, choices, answer_choice_string).
        def iter_examples(*, split: str) -> Iterator[tuple[str, list[str], str, str]]:
            split = str(split)
            if task == "hellaswag":
                ds_hs: Any = load_dataset("hellaswag", split=split)
                for ex in ds_hs:
                    exd = _as_dict(ex)
                    ctx_a = _norm_ws(exd.get("ctx_a", ""))
                    ctx_b = _norm_ws(exd.get("ctx_b", ""))
                    prompt = (ctx_a + " " + ctx_b).strip()
                    if prompt and not prompt.endswith((" ", "\n")):
                        prompt += " "
                    endings = cast(list[str], exd.get("endings", []))
                    endings = [str(e) for e in endings]
                    label = int(exd.get("label", -1))
                    if label < 0 or label >= len(endings):
                        continue
                    yield prompt, endings, str(endings[label]), split

            elif task == "piqa":
                ds_piqa: Any = load_dataset("piqa", split=split)
                for ex in ds_piqa:
                    exd = _as_dict(ex)
                    goal = _norm_ws(exd.get("goal", ""))
                    prompt = f"Goal: {goal}\nSolution:"
                    sol1 = str(exd.get("sol1", ""))
                    sol2 = str(exd.get("sol2", ""))
                    choices = [" " + _norm_ws(sol1), " " + _norm_ws(sol2)]
                    label = int(exd.get("label", -1))
                    if label not in (0, 1):
                        continue
                    yield prompt, choices, choices[label], split

            elif task == "winogrande":
                # Default to XL (standard).
                ds_wg: Any = load_dataset("winogrande", "winogrande_xl", split=split)
                for ex in ds_wg:
                    exd = _as_dict(ex)
                    sent = str(exd.get("sentence", ""))
                    if "_" not in sent:
                        continue
                    pre, post = sent.split("_", 1)
                    prompt = _norm_ws(pre).rstrip()
                    if prompt and not prompt.endswith((" ", "\n")):
                        prompt += " "
                    opt1 = _norm_ws(exd.get("option1", ""))
                    opt2 = _norm_ws(exd.get("option2", ""))
                    suffix = _norm_ws(post)
                    choices = [f"{opt1}{suffix}", f"{opt2}{suffix}"]
                    ans = str(exd.get("answer", "")).strip()
                    if ans == "1":
                        gold = choices[0]
                    elif ans == "2":
                        gold = choices[1]
                    else:
                        continue
                    yield prompt, choices, gold, split

            elif task in ("arc_easy", "arc_challenge"):
                subset = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
                ds_arc: Any = load_dataset("ai2_arc", subset, split=split)
                for ex in ds_arc:
                    exd = _as_dict(ex)
                    q = _norm_ws(exd.get("question", ""))
                    prompt = f"Question: {q}\nAnswer:"
                    ch = exd.get("choices", {})
                    labels = cast(list[str], ch.get("label", [])) if isinstance(ch, dict) else []
                    texts = cast(list[str], ch.get("text", [])) if isinstance(ch, dict) else []
                    if not labels or not texts or len(labels) != len(texts):
                        continue
                    # Use choice texts as completions (prefixed with space).
                    choices = [" " + _norm_ws(t) for t in texts]
                    key = str(exd.get("answerKey", "")).strip()
                    if not key:
                        continue
                    try:
                        idx = [str(x).strip() for x in labels].index(key)
                    except ValueError:
                        continue
                    gold = choices[int(idx)]
                    yield prompt, choices, gold, split

            elif task == "boolq":
                ds_bq: Any = load_dataset("boolq", split=split)
                for ex in ds_bq:
                    exd = _as_dict(ex)
                    passage = _norm_ws(exd.get("passage", ""))
                    question = _norm_ws(exd.get("question", ""))
                    prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
                    choices = [" yes", " no"]
                    ans = exd.get("answer", None)
                    if not isinstance(ans, (bool, int)):
                        continue
                    gold = choices[0] if bool(ans) else choices[1]
                    yield prompt, choices, gold, split

            else:
                raise ValueError(
                    f"Unsupported accuracy task {task!r}. "
                    "Supported: hellaswag, piqa, winogrande, arc_easy, arc_challenge, boolq."
                )

        # Few-shot prefix (best-effort): prepend N train examples as demonstrations.
        fewshot_prefix = ""
        if nshot > 0:
            try:
                k = 0
                for p, _choices, gold, _ in iter_examples(split="train"):
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
        sample_rows: list[list[str]] = []
        log_file = getattr(self.config, "log_file", None)
        log_lines: list[str] = []

        def _tr(s: str, mx: int = max_chars) -> str:
            ss = _norm_ws(str(s))
            return ss if len(ss) <= mx else ss[: max(0, mx - 1)] + "…"

        # Print header showing which model is being evaluated
        if stream_live:
            logger.console.print()
            logger.console.print(
                f"[highlight]━━━ {task} • {model_name} ━━━[/highlight]"
            )

        # Log file header
        if log_file:
            log_lines.append(f"{'=' * 80}")
            log_lines.append(f"TASK: {task} | MODEL: {model_name}")
            log_lines.append(f"{'=' * 80}")
            log_lines.append("")

        with torch.no_grad():
            for prompt, choices, gold, split in iter_examples(split="validation"):
                split_name = str(split)
                full_prompt = f"{fewshot_prefix}{prompt}"
                pred, scored = _pick_choice_with_scores(
                    model=model,
                    tokenizer=self.tokenizer,
                    prompt=full_prompt,
                    choices=list(choices),
                    device=self.device,
                    context_window=ctxw,
                )
                total += 1
                ok = str(pred) == str(gold)
                if ok:
                    correct += 1

                # Live streaming: print each example as it's evaluated
                if stream_live and (total % stream_every == 0):
                    acc_so_far = (correct / total) * 100.0 if total > 0 else 0.0
                    status = "[success]✓[/success]" if ok else "[error]✗[/error]"
                    # Compact single-line output for real-time feedback
                    logger.console.print(
                        f"  {status} [{total:>5}] acc={acc_so_far:5.1f}% │ "
                        f"[muted]prompt:[/muted] {_tr(prompt, 50)} │ "
                        f"[muted]pred:[/muted] {_tr(pred, 30)} │ "
                        f"[muted]gold:[/muted] {_tr(gold, 30)}"
                    )

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

                # Collect samples for end-of-task table (existing behavior)
                if nprint > 0 and shown < nprint:
                    if (not only_bad) or (not ok):
                        # Show best vs gold; also show runner-up score gap if available.
                        best_score = scored[0][1] if scored else float("nan")
                        gold_score = next((s for c, s in scored if str(c) == str(gold)), float("nan"))
                        sample_rows.append(
                            [
                                "✓" if ok else "✗",
                                _tr(prompt),
                                _tr(gold),
                                _tr(pred),
                                f"{(best_score - gold_score):.2f}" if (best_score == best_score and gold_score == gold_score) else "",
                            ]
                        )
                        shown += 1
                if limit is not None and total >= int(limit):
                    break

        acc = float(correct) / float(total) if total > 0 else 0.0

        # Final summary for live streaming
        if stream_live:
            logger.console.print()
            logger.console.print(
                f"  [highlight]━━━ {task} complete:[/highlight] "
                f"[metric]{correct}[/metric]/[metric]{total}[/metric] = "
                f"[success]{acc * 100.0:.2f}%[/success] accuracy"
            )
            logger.console.print()

        # Write log file (append mode to accumulate across models)
        if log_file and log_lines:
            log_lines.append(f"SUMMARY: {correct}/{total} = {acc * 100:.2f}%")
            log_lines.append("")
            log_lines.append("")

            log_path = Path(log_file)
            if self._output_dir and not log_path.is_absolute():
                log_path = self._output_dir / log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Append mode so both teacher and student results go to same file
            with log_path.open("a", encoding="utf-8") as f:
                f.write("\n".join(log_lines) + "\n")
            logger.info(f"Accuracy log appended to: {log_path}")

        logger.metric(str(model_name), acc * 100.0, "% acc")
        if sample_rows:
            logger.table(
                title=f"Accuracy examples • {task} • {model_name}",
                columns=["ok", "prompt", "gold", "pred", "Δ(best-gold)"],
                rows=sample_rows,
            )
        return TaskAccuracy(
            task=str(task),
            split=str(split_name),
            accuracy=float(acc),
            correct=int(correct),
            total=int(total),
        )

