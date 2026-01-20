"""Windowed completion log-probability scoring.

OPTIMIZED: Added batched scoring for multiple completions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LogprobCompletionWindowed:
    """Scores completion log-prob with a sliding context window.

    OPTIMIZED: Supports batched scoring via score_batch() when completions
    have similar lengths (batches context windows together).
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        context_window: int,
        valid_vocab_size: int | None = None,
    ) -> None:
        if int(context_window) <= 0:
            raise ValueError("context_window must be > 0")
        self.model = model
        self.device = device
        self.context_window = int(context_window)
        self.valid_vocab_size = int(valid_vocab_size) if valid_vocab_size is not None else None

    def score(self, *, prompt_ids: list[int], completion_ids: list[int]) -> float:
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if not completion_ids:
            raise ValueError("completion_ids must be non-empty")

        seq = list(prompt_ids) + list(completion_ids)
        total = 0.0
        start_k = len(prompt_ids)
        for k in range(start_k, len(seq)):
            ctx = seq[max(0, k - int(self.context_window)) : k]
            if not ctx:
                continue
            x = torch.tensor([ctx], device=self.device, dtype=torch.long)
            # IMPORTANT: evaluation only — ensure autograd is disabled.
            # This avoids issues with custom autograd kernels on some backends
            # (e.g. Metal) when upstream code uses inference tensors.
            with torch.no_grad():
                logits = self.model(x)
            v = logits[0, -1, :]
            if self.valid_vocab_size is not None:
                vv = int(self.valid_vocab_size)
                if int(v.shape[0]) < vv:
                    raise ValueError(
                        "Model returned logits with vocab smaller than valid_vocab_size "
                        f"(logits_vocab={int(v.shape[0])}, valid_vocab_size={vv})."
                    )
                if int(v.shape[0]) > vv:
                    v = v[:vv]
            lp = F.log_softmax(v, dim=-1)
            total += float(lp[int(seq[k])])
        return float(total)

    def score_batch(
        self, *, prompt_ids: list[int], completions_ids: list[list[int]]
    ) -> list[float]:
        """Score multiple completions with batched windowed scoring.

        OPTIMIZED: For short completions (common in MCQ), batches all sequences
        together when they fit within context window. Falls back to sequential
        for long completions.

        Args:
            prompt_ids: Token IDs for the prompt (shared across all completions)
            completions_ids: List of token ID lists, one per completion choice

        Returns:
            List of log-probability scores, one per completion
        """
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if not completions_ids:
            raise ValueError("completions_ids must be non-empty")

        # Check if all completions fit in a single forward pass
        # (prompt + max_completion <= context_window)
        max_comp_len = max(len(c) for c in completions_ids)
        prompt_len = len(prompt_ids)

        # If everything fits in context window, use efficient single-pass method
        if prompt_len + max_comp_len <= self.context_window:
            return self._score_batch_single_pass(prompt_ids, completions_ids)

        # Otherwise fall back to sequential scoring
        scores = []
        for comp_ids in completions_ids:
            if not comp_ids:
                scores.append(float("-inf"))
            else:
                scores.append(self.score(prompt_ids=prompt_ids, completion_ids=comp_ids))
        return scores

    def _score_batch_single_pass(
        self, prompt_ids: list[int], completions_ids: list[list[int]]
    ) -> list[float]:
        """Efficient batched scoring when all sequences fit in context window."""
        # Build all sequences: prompt + each completion
        all_seqs = []
        completion_lengths = []
        for comp_ids in completions_ids:
            if not comp_ids:
                all_seqs.append(list(prompt_ids) + [0])  # Placeholder
                completion_lengths.append(0)
            else:
                all_seqs.append(list(prompt_ids) + list(comp_ids))
                completion_lengths.append(len(comp_ids))

        # Pad to max length
        max_len = max(len(s) for s in all_seqs)
        padded_seqs = []
        for seq in all_seqs:
            padded = seq + [0] * (max_len - len(seq))
            padded_seqs.append(padded)

        # Single batched forward pass
        x = torch.tensor(padded_seqs, device=self.device, dtype=torch.long)
        with torch.no_grad():
            logits = self.model(x)
        if self.valid_vocab_size is not None:
            vv = int(self.valid_vocab_size)
            if int(logits.shape[-1]) < vv:
                raise ValueError(
                    "Model returned logits with vocab smaller than valid_vocab_size "
                    f"(logits_vocab={int(logits.shape[-1])}, valid_vocab_size={vv})."
                )
            if int(logits.shape[-1]) > vv:
                logits = logits[..., :vv]

        # Compute log probs
        logp = F.log_softmax(logits[:, :-1, :], dim=-1)  # [batch, seq_len-1, vocab]
        target = x[:, 1:]  # [batch, seq_len-1]

        # Extract score for each completion
        prompt_len = len(prompt_ids)
        scores = []
        for i, comp_len in enumerate(completion_lengths):
            if comp_len == 0:
                scores.append(float("-inf"))
                continue

            start = prompt_len - 1
            end = start + comp_len

            # Gather log probs for completion tokens
            tok_logp = logp[i, start:end, :].gather(
                dim=-1,
                index=target[i, start:end].unsqueeze(-1),
            )
            scores.append(float(tok_logp.sum().item()))

        return scores

