"""Full-sequence completion log-probability scoring.

OPTIMIZED: Added batched scoring for multiple completions in a single forward pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LogprobCompletionFullSequence:
    """Scores completion log-prob by a single forward pass over (prompt+completion).

    OPTIMIZED: Supports batched scoring via score_batch() for evaluating multiple
    completions in a single forward pass.
    """

    def __init__(
        self, *, model: nn.Module, device: torch.device, valid_vocab_size: int | None = None
    ) -> None:
        self.model = model
        self.device = device
        self.valid_vocab_size = int(valid_vocab_size) if valid_vocab_size is not None else None

    def score(self, *, prompt_ids: list[int], completion_ids: list[int]) -> float:
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if not completion_ids:
            raise ValueError("completion_ids must be non-empty")

        seq = list(prompt_ids) + list(completion_ids)
        x = torch.tensor([seq], device=self.device, dtype=torch.long)
        # IMPORTANT: evaluation only — ensure autograd is disabled.
        with torch.no_grad():
            logits = self.model(x)
        if logits.ndim != 3:
            raise ValueError(f"Expected logits (B,T,V), got {tuple(logits.shape)}")
        if int(logits.shape[1]) != len(seq):
            raise ValueError("Unexpected logits length mismatch")
        if self.valid_vocab_size is not None:
            vv = int(self.valid_vocab_size)
            if int(logits.shape[-1]) < vv:
                raise ValueError(
                    "Model returned logits with vocab smaller than valid_vocab_size "
                    f"(logits_vocab={int(logits.shape[-1])}, valid_vocab_size={vv})."
                )
            if int(logits.shape[-1]) > vv:
                logits = logits[..., :vv]

        logp = F.log_softmax(logits[:, :-1, :], dim=-1)
        target = x[:, 1:]

        start = len(prompt_ids) - 1
        end = start + len(completion_ids)
        tok_logp = logp[0, start:end, :].gather(
            dim=-1,
            index=target[0, start:end].unsqueeze(-1),
        )
        return float(tok_logp.sum().item())

    def score_batch(
        self, *, prompt_ids: list[int], completions_ids: list[list[int]]
    ) -> list[float]:
        """Score multiple completions in a single batched forward pass.

        OPTIMIZED: Batches all (prompt + completion_i) sequences together,
        runs ONE forward pass, then extracts scores for each completion.

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

        # Build all sequences: prompt + each completion
        all_seqs = []
        completion_lengths = []
        for comp_ids in completions_ids:
            if not comp_ids:
                raise ValueError("Each completion must be non-empty")
            seq = list(prompt_ids) + list(comp_ids)
            all_seqs.append(seq)
            completion_lengths.append(len(comp_ids))

        # Pad to max length for batching
        max_len = max(len(s) for s in all_seqs)
        padded_seqs = []
        for seq in all_seqs:
            # Pad with 0s (will be masked out in scoring)
            padded = seq + [0] * (max_len - len(seq))
            padded_seqs.append(padded)

        # Single batched forward pass
        x = torch.tensor(padded_seqs, device=self.device, dtype=torch.long)
        with torch.no_grad():
            logits = self.model(x)

        if logits.ndim != 3:
            raise ValueError(f"Expected logits (B,T,V), got {tuple(logits.shape)}")
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
            start = prompt_len - 1
            end = start + comp_len

            # Gather log probs for completion tokens
            tok_logp = logp[i, start:end, :].gather(
                dim=-1,
                index=target[i, start:end].unsqueeze(-1),
            )
            scores.append(float(tok_logp.sum().item()))

        return scores

