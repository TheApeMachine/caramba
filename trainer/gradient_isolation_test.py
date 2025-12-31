from __future__ import annotations

from collections.abc import Iterator
from typing import Tuple

import torch
from torch import nn

from caramba.trainer.gradient_isolation import apply_trainable_scope


def test_apply_trainable_scope_freezes_and_unfreezes_by_regex() -> None:
    class Sys(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.module = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
            return self.module(x)

        def named_parameters(self) -> Iterator[Tuple[str, torch.Tensor]]:
            return self.module.named_parameters()

    s = Sys()
    stats = apply_trainable_scope(s, trainable=[r"2\.weight$", r"2\.bias$"])
    assert stats["trainable"] == 2
    assert stats["total"] >= 2

    for name, p in s.named_parameters():
        if name.endswith(("2.weight", "2.bias")):
            assert p.requires_grad is True
        else:
            assert p.requires_grad is False

