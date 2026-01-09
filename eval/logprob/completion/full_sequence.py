"""Full-sequence completion log-probability scoring."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LogprobCompletionFullSequence:
    """Scores completion log-prob by a single forward pass over (prompt+completion)."""

    def __init__(self, *, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    def score(self, *, prompt_ids: list[int], completion_ids: list[int]) -> float:
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if not completion_ids:
            raise ValueError("completion_ids must be non-empty")

        seq = list(prompt_ids) + list(completion_ids)
        x = torch.tensor([seq], device=self.device, dtype=torch.long)
        logits = self.model(x)
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
        return float(tok_logp.sum().item())

