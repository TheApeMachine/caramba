from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from benchmark.behavior.types import ChoiceLogprobDetails
from data.tokenizers.builder import TokenizerBuilder

from eval.logprob.completion.full_sequence import LogprobCompletionFullSequence
from eval.logprob.completion.windowed import LogprobCompletionWindowed


def build_tokenizer(tokenizer_config) -> Any:
    return TokenizerBuilder().build(tokenizer_config)


def tokenizer_vocab_size(tok: Any) -> int | None:
    # TokenizerBuilder outputs wrappers; tiktoken wrapper stores _enc.n_vocab.
    for attr in ("n_vocab", "vocab_size"):
        if hasattr(tok, attr):
            try:
                return int(getattr(tok, attr))
            except Exception:
                pass
    if hasattr(tok, "_enc"):
        try:
            return int(getattr(getattr(tok, "_enc"), "n_vocab"))
        except Exception:
            return None
    return None


def softmax(xs: list[float]) -> list[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(float(x) - float(m)) for x in xs]
    s = sum(exps)
    if not (s > 0.0):
        raise RuntimeError("softmax: sum(exp) is not positive.")
    return [float(e) / float(s) for e in exps]


def score_choices_by_logprob(
    *,
    model: nn.Module,
    device: torch.device,
    prompt_ids: list[int],
    choices: list[str],
    correct_index: int,
    tokenizer: Any,
    context_window: int | None,
    valid_vocab_size: int | None,
) -> ChoiceLogprobDetails:
    if not choices:
        raise ValueError("choice_logprob: empty choices.")
    if not (0 <= int(correct_index) < len(choices)):
        raise ValueError(f"choice_logprob: correct_index out of range: {correct_index}.")

    completion = (
        LogprobCompletionWindowed(
            model=model,
            device=device,
            context_window=int(context_window),
            valid_vocab_size=(int(valid_vocab_size) if valid_vocab_size is not None else None),
        )
        if context_window is not None
        else LogprobCompletionFullSequence(
            model=model,
            device=device,
            valid_vocab_size=(int(valid_vocab_size) if valid_vocab_size is not None else None),
        )
    )

    choice_ids = [tokenizer.encode(str(c)) for c in choices]
    scores = completion.score_batch(prompt_ids=list(prompt_ids), completions_ids=choice_ids)
    logps = [float(s) for s in scores]
    probs = softmax(logps)

    picked_index = max(range(len(logps)), key=lambda i: float(logps[i]))
    picked = str(choices[int(picked_index)])

    correct = str(choices[int(correct_index)])
    correct_lp = float(logps[int(correct_index)])
    best_other = max(float(lp) for i, lp in enumerate(logps) if i != int(correct_index))
    margin = float(correct_lp - best_other)

    return ChoiceLogprobDetails(
        choices=[str(c) for c in choices],
        logprobs=logps,
        probs=probs,
        picked=picked,
        picked_index=int(picked_index),
        correct=correct,
        correct_index=int(correct_index),
        margin_logprob=margin,
    )


def generate_greedy(
    *,
    model: nn.Module,
    device: torch.device,
    prompt_ids: list[int],
    tokenizer: Any,
    max_new_tokens: int,
    context_window: int | None,
    valid_vocab_size: int | None,
) -> str:
    """Strict greedy generation using the KV-cache Generator.

    IMPORTANT: No fallback behavior. If generation fails, we raise.
    """
    from infer.generate import Generator, GenerateConfig, sample_next_token

    if max_new_tokens <= 0:
        raise ValueError("generate_greedy: max_new_tokens must be > 0.")

    input_ids = torch.tensor([list(prompt_ids)], device=device)
    gen_config = GenerateConfig(
        max_new_tokens=int(max_new_tokens),
        temperature=0.0,
        max_seq_len=(int(context_window) if context_window is not None else 2048),
    )
    generator = Generator(model, config=gen_config, device=device)
    vv = int(valid_vocab_size) if valid_vocab_size is not None else None

    generated: list[int] = []
    with torch.no_grad():
        logits = generator.prefill(input_ids)
        for _ in range(int(max_new_tokens)):
            if vv is not None and int(getattr(logits, "shape", [0])[-1]) > int(vv):
                logits = logits[..., : int(vv)]
            next_token = sample_next_token(logits, temperature=0.0)
            # NOTE: We do not treat token id 0 as EOS universally; tokenizers differ.
            # Stop conditions should be expressed via match rules / max_new_tokens only.
            generated.append(int(next_token.item()))

            logits = generator.decode_step(next_token)
            if context_window is not None and (len(prompt_ids) + len(generated)) >= int(context_window):
                break

    return str(tokenizer.decode(generated))


def score_expected_logprob(
    *,
    model: nn.Module,
    device: torch.device,
    prompt_ids: list[int],
    expected_text: str,
    tokenizer: Any,
    context_window: int | None,
    valid_vocab_size: int | None,
) -> float:
    """Compute logprob(expected | prompt) as a raw confidence signal."""
    completion = (
        LogprobCompletionWindowed(
            model=model,
            device=device,
            context_window=int(context_window),
            valid_vocab_size=(int(valid_vocab_size) if valid_vocab_size is not None else None),
        )
        if context_window is not None
        else LogprobCompletionFullSequence(
            model=model,
            device=device,
            valid_vocab_size=(int(valid_vocab_size) if valid_vocab_size is not None else None),
        )
    )
    exp_ids = tokenizer.encode(str(expected_text))
    # score_batch expects list of completions
    score = completion.score_batch(prompt_ids=list(prompt_ids), completions_ids=[exp_ids])[0]
    return float(score)

