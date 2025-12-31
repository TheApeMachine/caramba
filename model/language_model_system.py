"""Language model system.

This wraps the existing `model.Model` into a generic "system" shape:
- accepts dict-like batches
- returns dict-like outputs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from caramba.console import logger

from caramba.compiler.lower import Lowerer
from caramba.compiler.validate import Validator
from caramba.config.model import ModelConfig
from caramba.model import Model


@dataclass
class LanguageModelSystem:
    """A system that produces next-token logits from token batches.

    Config:
    - model: ModelConfig payload (required)
    """

    model: dict[str, Any]

    def __post_init__(self) -> None:
        cfg = ModelConfig.model_validate(self.model)
        cfg = Lowerer().lower_model(cfg)
        Validator().validate_model_config(cfg)
        self.config = cfg
        self.module: nn.Module = Model(cfg)

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "LanguageModelSystem":
        self.module = self.module.to(device=device, dtype=dtype)
        return self

    def train(self, mode: bool = True) -> None:
        self.module.train(mode)

    def eval(self) -> None:
        self.module.eval()

    def forward(self, batch: dict[str, Any], *, ctx: object | None = None) -> dict[str, Any]:
        input_ids: Tensor = batch["input_ids"]
        # Return features if supported (needed for diffusion head training).
        try:
            result = self.module(input_ids, ctx=ctx, return_features=True)  # type: ignore[call-arg]
            if isinstance(result, tuple) and len(result) == 2:
                features, logits = result
                out = {"features": features, "logits": logits, "_system": self.module}
                # Best-effort: attach MOSAIC aux outputs when present on ctx.
                aux = getattr(ctx, "mosaic_aux_out", None) if ctx is not None else None
                if isinstance(aux, dict):
                    for k, v in aux.items():
                        if isinstance(k, str) and isinstance(v, Tensor):
                            out[k] = v
                return out
        except TypeError:
            logger.error("Failed to forward, continuing")
        logits = self.module(input_ids, ctx=ctx)  # type: ignore[call-arg]
        out2 = {"logits": logits, "_system": self.module}
        aux2 = getattr(ctx, "mosaic_aux_out", None) if ctx is not None else None
        if isinstance(aux2, dict):
            for k, v in aux2.items():
                if isinstance(k, str) and isinstance(v, Tensor):
                    out2[k] = v
        return out2

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.module.load_state_dict(state)

