from __future__ import annotations

from torch import nn

from trainer.gradient_isolation import apply_trainable_scope


def test_apply_trainable_scope_freezes_and_unfreezes_by_regex() -> None:
    class Sys:
        def __init__(self) -> None:
            self.module = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))

        def named_parameters(self):
            return self.module.named_parameters()

    s = Sys()
    stats = apply_trainable_scope(s, trainable=[r"2\.weight$", r"2\.bias$"])
    assert stats["trainable"] == 2
    assert stats["total"] >= 2

    for name, p in s.named_parameters():
        if name.endswith("2.weight") or name.endswith("2.bias"):
            assert p.requires_grad is True
        else:
            assert p.requires_grad is False

