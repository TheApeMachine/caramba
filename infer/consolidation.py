"""Idle-time consolidation using replay windows (and optional online learning)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from caramba.infer.event_runtime import StreamModelRunner
from caramba.infer.replay import ReplayBuffer
from caramba.trainer.online import OnlineLearner


@dataclass(slots=True)
class ReplayConsolidator:
    runner: StreamModelRunner
    replay: ReplayBuffer
    block_size: int = 256
    batch_size: int = 2
    online: OnlineLearner | None = None

    def consolidate_once(self) -> dict[str, float] | None:
        if self.replay.size() <= 0:
            return None
        try:
            device = next(self.runner.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        x, y = self.replay.sample_next_token_batch(
            batch_size=int(self.batch_size),
            block_size=int(self.block_size),
            device=device,
        )

        # Forward without advancing the public "pos" counter.
        logits, _aux = self.runner.forward_chunk(x, advance_pos=False)
        # logits: (B,T,V)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        metrics = {"replay/loss": float(loss.detach().item())}

        if self.online is not None:
            upd = self.online.step(loss)
            metrics.update({f"online/{k}": float(v) for k, v in upd.items()})

        return metrics

