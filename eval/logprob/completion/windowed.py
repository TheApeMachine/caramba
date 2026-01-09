"""Windowed completion log-probability scoring."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LogprobCompletionWindowed:
    """Scores completion log-prob with a sliding context window."""

    def __init__(
        self, *, model: nn.Module, device: torch.device, context_window: int
    ) -> None:
        if int(context_window) <= 0:
            raise ValueError("context_window must be > 0")
        self.model = model
        self.device = device
        self.context_window = int(context_window)

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
            logits = self.model(x)
            lp = F.log_softmax(logits[0, -1, :], dim=-1)
            total += float(lp[int(seq[k])])
        return float(total)

