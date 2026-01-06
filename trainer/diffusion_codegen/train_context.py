"""Training context for diffusion codegen

Encapsulates state shared across the training loop (model, tokenizer, schedule,
scaler, counters).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer
from torch import Tensor
from torch.cuda.amp import GradScaler

from caramba.diffusion.schedule import NoiseSchedule


@dataclass
class TrainingContext:
    """Shared training state."""

    device: torch.device
    model: Any
    tokenizer: Tokenizer
    pad_id: int
    alpha_bar: Tensor
    schedule: NoiseSchedule
    scaler: GradScaler
    hidden_size: int
    seq_len: int
    step_counter: int = 0

    @classmethod
    def fromTokenizerFile(
        cls,
        *,
        model: Any,
        tokenizer_file: str,
        alpha_bar: Tensor,
        schedule: NoiseSchedule,
    ) -> "TrainingContext":
        device = next(model.parameters()).device
        tokenizer = Tokenizer.from_file(str(Path(tokenizer_file)))
        pad_id = tokenizer.token_to_id("<pad>")
        if pad_id is None:
            raise ValueError("Tokenizer must define <pad>.")
        return cls(
            device=device,
            model=model,
            tokenizer=tokenizer,
            pad_id=int(pad_id),
            alpha_bar=alpha_bar,
            schedule=schedule,
            scaler=GradScaler(enabled=(device.type == "cuda")),
            hidden_size=int(getattr(model, "hidden_size", 512)),
            seq_len=int(getattr(model, "max_len", 128)),
        )

