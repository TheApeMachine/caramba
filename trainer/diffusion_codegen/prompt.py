"""Prompt utilities

Encodes text prompts to model prompt embeddings and masks using a Tokenizer and
the model's embedding table.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from data.tokenizers.training import TrainingTokenizer


@dataclass(frozen=True, slots=True)
class PromptEncoder:
    """Prompt encoder

    Converts prompt text into (prompt_emb, prompt_pad_mask) aligned to seq_len.
    """

    tokenizer: TrainingTokenizer
    embedding: nn.Embedding
    pad_id: int
    seq_len: int

    def encode(self, *, prompt: str, device: torch.device) -> tuple[Tensor, Tensor]:
        """Encode prompt text as embeddings and padding mask."""

        ids = self.tokenizer.encode(prompt).ids
        ids = self.padOrTruncate(ids=ids)
        tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        pad_mask = tokens.eq(int(self.pad_id))
        emb = self.embedding(tokens)
        return emb, pad_mask

    def padOrTruncate(self, *, ids: list[int]) -> list[int]:
        if len(ids) >= int(self.seq_len):
            return ids[: int(self.seq_len)]
        return ids + [int(self.pad_id)] * (int(self.seq_len) - len(ids))

