"""Medium-timescale online learning (adapter/LoRA-only updates).

This is intended for *idle-time* or *budgeted* updates during inference:
- Only a small parameter subset is updated (LoRA/adapters), never the full model.
- A caller supplies a loss tensor computed from replay data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


def select_adapter_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Select adapter/LoRA parameters from a model.

    Current heuristic:
    - any parameter whose name includes 'lora_' (covers LoRALinearLayer.lora_A/B)
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not isinstance(p, nn.Parameter):
            continue
        if "lora_" in str(name):
            params.append(p)
    return params


@dataclass(slots=True)
class OnlineLearner:
    """Adapter-only optimizer wrapper."""

    model: nn.Module
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    params: list[nn.Parameter] = field(default_factory=list)
    optimizer: torch.optim.Optimizer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.params:
            self.params = select_adapter_parameters(self.model)
        if not self.params:
            raise ValueError("OnlineLearner has no adapter params to update (no 'lora_' params found).")

        self.optimizer = torch.optim.AdamW(
            self.params,
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )

    def step(self, loss: Tensor) -> dict[str, float]:
        if not isinstance(loss, Tensor):
            raise TypeError(f"loss must be a Tensor, got {type(loss).__name__}")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = 0.0
        if self.grad_clip is not None:
            grad_norm_t = torch.nn.utils.clip_grad_norm_(self.params, float(self.grad_clip))
            grad_norm = float(grad_norm_t.detach().float().item())

        self.optimizer.step()
        return {"grad_norm": float(grad_norm)}

