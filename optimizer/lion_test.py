from __future__ import annotations

import pytest
import torch

from caramba.optimizer.lion import Lion


def test_lion_step_updates_parameter_and_supports_closure() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    opt = Lion([p], lr=0.1, betas=(0.0, 0.0), weight_decay=0.0, fused=False)

    def closure() -> torch.Tensor:
        # Fake loss: set gradient deterministically.
        if p.grad is not None:
            p.grad.zero_()
        p.grad = torch.tensor([3.0, -4.0])
        return (p.sum() * 0.0) + 1.0

    loss = opt.step(closure)
    assert loss is not None
    assert float(loss) == 1.0

    # With beta1=0, m becomes grad. Update is p -= lr*sign(m).
    # sign([3,-4]) = [1,-1] => p = [1-0.1, -2+0.1]
    assert torch.allclose(p.data, torch.tensor([0.9, -1.9]))


def test_lion_step_applies_weight_decay() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0]))
    p.grad = torch.tensor([1.0])
    opt = Lion([p], lr=0.1, betas=(0.0, 0.0), weight_decay=0.5, fused=False)
    opt.step()

    # decay: p *= (1 - lr*wd) = 0.95, then p -= lr*sign(grad) = 0.1
    assert torch.allclose(p.data, torch.tensor([0.85]))

