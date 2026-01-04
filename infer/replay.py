"""Replay buffer for idle-time consolidation and online learning."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(slots=True)
class ReplayBuffer:
    """A tiny sequence replay buffer.

    Stores sequences of token IDs (1D) and samples random windows for next-token training.
    """

    max_sequences: int = 1024
    rng_seed: int = 1337
    _seqs: list[list[int]] = field(default_factory=list)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.max_sequences = int(self.max_sequences)
        if self.max_sequences < 1:
            raise ValueError(f"max_sequences must be >= 1, got {self.max_sequences}")
        self._rng = random.Random(int(self.rng_seed))

    def add(self, seq: Tensor | list[int]) -> None:
        if isinstance(seq, Tensor):
            if seq.ndim != 1:
                raise ValueError(f"ReplayBuffer.add expects 1D Tensor, got {tuple(seq.shape)}")
            seq_list = [int(x) for x in seq.detach().to(dtype=torch.long).cpu().tolist()]
        else:
            seq_list = [int(x) for x in seq]
        if len(seq_list) < 2:
            return
        self._seqs.append(seq_list)
        if len(self._seqs) > int(self.max_sequences):
            # Drop oldest.
            self._seqs = self._seqs[-int(self.max_sequences) :]

    def size(self) -> int:
        return len(self._seqs)

    def sample_next_token_batch(
        self,
        *,
        batch_size: int,
        block_size: int,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample (input_ids, target_ids) each with shape (B, T)."""
        B = int(batch_size)
        T = int(block_size)
        if B < 1:
            raise ValueError(f"batch_size must be >= 1, got {B}")
        if T < 2:
            raise ValueError(f"block_size must be >= 2, got {T}")
        if not self._seqs:
            raise RuntimeError("ReplayBuffer is empty")

        xs: list[list[int]] = []
        ys: list[list[int]] = []
        for _ in range(B):
            seq = self._rng.choice(self._seqs)
            if len(seq) < T + 1:
                # Pad by sampling another; worst-case, truncate minimal.
                start = 0
            else:
                start = self._rng.randrange(0, len(seq) - (T + 1) + 1)
            window = seq[start : start + T + 1]
            if len(window) < T + 1:
                # Right-pad by repeating last token (cheap, avoids shape issues).
                last = window[-1] if window else 0
                window = window + [last] * (T + 1 - len(window))
            x = window[:-1]
            y = window[1:]
            xs.append(x)
            ys.append(y)

        x_t = torch.tensor(xs, dtype=torch.long, device=device)
        y_t = torch.tensor(ys, dtype=torch.long, device=device)
        return x_t, y_t

