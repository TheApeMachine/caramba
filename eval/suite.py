"""Behavioral evaluation for teacher/student comparison.

After upcycling, we need to verify the student model behaves correctly—not
just that its internal activations match the teacher. This module runs
behavioral tests: prompt the model, check if it produces the right answer.

Evaluation modes:
- choice_logprob: Pick the most likely choice from a list (log-prob scoring)
- int_greedy: Generate greedily and extract the first integer
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from torch import nn

from config.eval import EvalCase, EvalThresholds, EvalVerifyConfig
from data.tokenizers.base import Tokenizer
from data.tokenizers.builder import TokenizerBuilder


def _tokenizer_vocab_size(tokenizer: Tokenizer) -> int | None:
    """Best-effort tokenizer vocab size.

    Used to mask padded model vocab slots (e.g. 50304 vs 50257) so scoring and
    greedy generation operate over the real token space.
    """
    for attr in ("n_vocab", "vocab_size"):
        if hasattr(tokenizer, attr):
            try:
                return int(getattr(tokenizer, attr))
            except Exception:
                pass
    if hasattr(tokenizer, "_enc"):
        try:
            return int(getattr(getattr(tokenizer, "_enc"), "n_vocab"))
        except Exception:
            return None
    return None


class EvalCaseResult:
    """Records per-case outcomes for teacher and student.

    Stores whether each model got the answer right and what they produced.
    """

    def __init__(
        self,
        *,
        case_id: str,
        teacher_ok: bool,
        student_ok: bool,
        teacher_answer: str,
        student_answer: str,
    ) -> None:
        """Initialize the case result."""
        self.case_id: str = str(case_id)
        self.teacher_ok: bool = bool(teacher_ok)
        self.student_ok: bool = bool(student_ok)
        self.teacher_answer: str = str(teacher_answer)
        self.student_answer: str = str(student_answer)


class EvalSummary:
    """Aggregates evaluation metrics across all test cases.

    Computes accuracy for both teacher and student, enabling comparison
    to detect quality regressions from upcycling.
    """

    def __init__(self, *, results: list[EvalCaseResult]) -> None:
        """Compute aggregate metrics from individual results."""
        if not results:
            raise ValueError("results must be non-empty")
        self.results: list[EvalCaseResult] = results
        self.teacher_accuracy: float = (
            sum(1 for r in results if r.teacher_ok) / float(len(results))
        )
        self.student_accuracy: float = (
            sum(1 for r in results if r.student_ok) / float(len(results))
        )


def run_eval_verify(
    *,
    teacher: nn.Module,
    student: nn.Module,
    cfg: EvalVerifyConfig,
    device: torch.device,
) -> EvalSummary:
    """Run the configured evaluation suite.

    Runs all test cases against both teacher and student models,
    returning an aggregate summary of their performance.
    """
    tokenizer = TokenizerBuilder().build(cfg.tokenizer)
    teacher.eval()
    student.eval()

    results: list[EvalCaseResult] = []
    with torch.no_grad():
        for case in list(cfg.cases):
            results.append(
                _run_case(
                    teacher=teacher,
                    student=student,
                    case=case,
                    tokenizer=tokenizer,
                    max_new_tokens=int(cfg.max_new_tokens),
                    context_window=cfg.context_window,
                    device=device,
                )
            )
    return EvalSummary(results=results)


def assert_eval_thresholds(
    *, summary: EvalSummary, thresholds: EvalThresholds
) -> None:
    """Validate evaluation metrics against thresholds.

    Raises ValueError if the student accuracy is too low or the
    accuracy drop from teacher to student is too large.
    """
    if summary.student_accuracy < float(thresholds.min_student_accuracy):
        raise ValueError(
            "eval failed: student accuracy below threshold: "
            f"acc={summary.student_accuracy:.3f}, "
            f"min={float(thresholds.min_student_accuracy):.3f}"
        )
    drop = float(summary.teacher_accuracy - summary.student_accuracy)
    if drop > float(thresholds.max_accuracy_drop):
        raise ValueError(
            "eval failed: accuracy drop exceeded threshold: "
            f"teacher={summary.teacher_accuracy:.3f}, "
            f"student={summary.student_accuracy:.3f}, "
            f"drop={drop:.3f}, "
            f"max_drop={float(thresholds.max_accuracy_drop):.3f}"
        )


def _run_case(
    *,
    teacher: nn.Module,
    student: nn.Module,
    case: EvalCase,
    tokenizer: Tokenizer,
    max_new_tokens: int,
    context_window: int | None,
    device: torch.device,
) -> EvalCaseResult:
    """Run a single evaluation case."""
    prompt_ids = tokenizer.encode(case.prompt)
    if not prompt_ids:
        raise ValueError(f"Case {case.id!r} encoded to empty prompt ids")

    match case.kind:
        case "choice_logprob":
            if not isinstance(case.answer, str):
                raise TypeError(
                    f"Case {case.id!r}: expected answer to be str for choice_logprob, "
                    f"got {type(case.answer).__name__}"
                )
            if case.choices is None:
                raise ValueError(
                    f"Case {case.id!r}: choices must not be None for choice_logprob"
                )
            t_choice = _pick_choice_by_logprob(
                model=teacher,
                prompt_ids=prompt_ids,
                choices=case.choices,
                tokenizer=tokenizer,
                device=device,
                context_window=context_window,
            )
            s_choice = _pick_choice_by_logprob(
                model=student,
                prompt_ids=prompt_ids,
                choices=case.choices,
                tokenizer=tokenizer,
                device=device,
                context_window=context_window,
            )
            return EvalCaseResult(
                case_id=case.id,
                teacher_ok=(t_choice == case.answer),
                student_ok=(s_choice == case.answer),
                teacher_answer=t_choice,
                student_answer=s_choice,
            )
        case "int_greedy":
            if not isinstance(case.answer, int):
                raise TypeError(
                    f"Case {case.id!r}: expected answer to be int for int_greedy, "
                    f"got {type(case.answer).__name__}"
                )
            t_text = _greedy_generate(
                model=teacher,
                prompt_ids=prompt_ids,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
                context_window=context_window,
            )
            s_text = _greedy_generate(
                model=student,
                prompt_ids=prompt_ids,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
                context_window=context_window,
            )
            t_int = _extract_first_int(t_text)
            s_int = _extract_first_int(s_text)
            return EvalCaseResult(
                case_id=case.id,
                teacher_ok=(t_int == case.answer),
                student_ok=(s_int == case.answer),
                teacher_answer=str(t_int),
                student_answer=str(s_int),
            )
        case "float_greedy":
            if not isinstance(case.answer, (int, float)):
                raise TypeError(
                    f"Case {case.id!r}: expected answer to be int or float for float_greedy, "
                    f"got {type(case.answer).__name__}"
                )
            t_text = _greedy_generate(
                model=teacher,
                prompt_ids=prompt_ids,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
                context_window=context_window,
            )
            s_text = _greedy_generate(
                model=student,
                prompt_ids=prompt_ids,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
                context_window=context_window,
            )
            t_float = _extract_first_float(t_text)
            s_float = _extract_first_float(s_text)
            return EvalCaseResult(
                case_id=case.id,
                teacher_ok=(abs(t_float - float(case.answer)) < 1e-6),
                student_ok=(abs(s_float - float(case.answer)) < 1e-6),
                teacher_answer=str(t_float),
                student_answer=str(s_float),
            )
        case "exact_match_greedy":
            if not isinstance(case.answer, str):
                raise TypeError(
                    f"Case {case.id!r}: expected answer to be str for exact_match_greedy, "
                    f"got {type(case.answer).__name__}"
                )
            t_text = _greedy_generate(
                model=teacher,
                prompt_ids=prompt_ids,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
                context_window=context_window,
            )
            s_text = _greedy_generate(
                model=student,
                prompt_ids=prompt_ids,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
                context_window=context_window,
            )
            expected = case.answer.strip()

            def _canon(out_text: str) -> str:
                mode = str(getattr(case, "match", "exact")).strip().lower()
                raw = out_text
                if mode == "first_line":
                    raw = raw.splitlines()[0] if raw.splitlines() else raw
                # For "prefix" we still return the full stripped output; comparison uses startswith.
                return raw.strip()

            t_out = t_text.strip()
            s_out = s_text.strip()
            t_out = _canon(t_out)
            s_out = _canon(s_out)

            match_mode = str(getattr(case, "match", "exact")).strip().lower()
            if match_mode == "prefix":
                t_ok = t_out.startswith(expected)
                s_ok = s_out.startswith(expected)
            else:
                t_ok = (t_out == expected)
                s_ok = (s_out == expected)
            return EvalCaseResult(
                case_id=case.id,
                teacher_ok=bool(t_ok),
                student_ok=bool(s_ok),
                teacher_answer=t_out,
                student_answer=s_out,
            )
        case _:
            raise ValueError(f"Unsupported eval kind: {case.kind!r}")


def _extract_first_int(text: str) -> int:
    """Extract the first integer from text, defaulting to 0."""
    m = re.search(r"-?\d+", str(text))
    if m is None:
        return 0
    return int(m.group(0))


def _extract_first_float(text: str) -> float:
    """Extract the first float from text, defaulting to 0.0."""
    m = re.search(r"-?\d+(?:\.\d+)?", str(text))
    if m is None:
        return 0.0
    return float(m.group(0))


def _pick_choice_by_logprob(
    *,
    model: nn.Module,
    prompt_ids: list[int],
    choices: list[str],
    tokenizer: Tokenizer,
    device: torch.device,
    context_window: int | None,
) -> str:
    """Pick the choice with highest log-probability continuation."""
    if not choices:
        raise ValueError("choices must be non-empty")
    best: tuple[float, str] | None = None
    for choice in choices:
        score = _score_completion_logprob(
            model=model,
            prompt_ids=prompt_ids,
            completion_ids=tokenizer.encode(str(choice)),
            tokenizer=tokenizer,
            device=device,
            context_window=context_window,
        )
        item = (float(score), str(choice))
        best = item if best is None or item[0] > best[0] else best
    if best is None:
        raise RuntimeError("No choices were scored")
    return best[1]


def _score_completion_logprob(
    *,
    model: nn.Module,
    prompt_ids: list[int],
    completion_ids: list[int],
    tokenizer: Tokenizer,
    device: torch.device,
    context_window: int | None,
) -> float:
    """Score a completion by summing token log-probabilities."""
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")
    if not completion_ids:
        raise ValueError("completion_ids must be non-empty")

    if context_window is not None:
        return _score_completion_logprob_windowed(
            model=model,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            tokenizer=tokenizer,
            device=device,
            context_window=int(context_window),
        )

    seq = prompt_ids + completion_ids
    x = torch.tensor([seq], device=device, dtype=torch.long)
    logits = model(x)
    if logits.ndim != 3:
        raise ValueError(f"Expected logits (B,T,V), got {logits.shape}")
    if int(logits.shape[1]) != len(seq):
        raise ValueError("Unexpected logits length mismatch")

    vv = _tokenizer_vocab_size(tokenizer)
    if vv is not None:
        if int(logits.shape[-1]) < int(vv):
            raise ValueError(
                "Model returned logits with vocab smaller than tokenizer vocab "
                f"(logits_vocab={int(logits.shape[-1])}, tokenizer_vocab={int(vv)})."
            )
        if int(logits.shape[-1]) > int(vv):
            logits = logits[..., : int(vv)]

    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = x[:, 1:]

    start = len(prompt_ids) - 1
    end = start + len(completion_ids)
    tok_logp = logp[0, start:end, :].gather(
        dim=-1,
        index=target[0, start:end].unsqueeze(-1),
    )
    return float(tok_logp.sum())


def _score_completion_logprob_windowed(
    *,
    model: nn.Module,
    prompt_ids: list[int],
    completion_ids: list[int],
    tokenizer: Tokenizer,
    device: torch.device,
    context_window: int,
) -> float:
    """Score completion with sliding window for long contexts."""
    if context_window <= 0:
        raise ValueError("context_window must be > 0")

    seq = prompt_ids + completion_ids
    total = 0.0
    start_k = len(prompt_ids)
    vv = _tokenizer_vocab_size(tokenizer)
    for k in range(start_k, len(seq)):
        ctx = seq[max(0, k - context_window) : k]
        if not ctx:
            raise RuntimeError("Empty context during windowed scoring")
        x = torch.tensor([ctx], device=device, dtype=torch.long)
        logits = model(x)
        next_id = int(seq[k])
        v = logits[0, -1, :]
        if vv is not None and int(v.shape[0]) < int(vv):
            raise ValueError(
                "Model returned logits with vocab smaller than tokenizer vocab "
                f"(logits_vocab={int(v.shape[0])}, tokenizer_vocab={int(vv)})."
            )
        if vv is not None and int(v.shape[0]) > int(vv):
            v = v[: int(vv)]
        lp = F.log_softmax(v, dim=-1)
        total += float(lp[next_id])
    return float(total)


def _greedy_generate(
    *,
    model: nn.Module,
    prompt_ids: list[int],
    tokenizer: Tokenizer,
    device: torch.device,
    max_new_tokens: int,
    context_window: int | None,
) -> str:
    """Generate tokens greedily (argmax at each step)."""
    ids = list(prompt_ids)
    vv = _tokenizer_vocab_size(tokenizer)
    for _ in range(int(max_new_tokens)):
        ctx = ids
        if context_window is not None:
            ctx = ids[-int(context_window) :]
        x = torch.tensor([ctx], device=device, dtype=torch.long)
        logits = model(x)
        v = logits[0, -1, :]
        if vv is not None and int(v.shape[0]) > int(vv):
            v = v[: int(vv)]
        next_id = int(torch.argmax(v).item())
        ids.append(next_id)
    return tokenizer.decode(ids[len(prompt_ids) :])
