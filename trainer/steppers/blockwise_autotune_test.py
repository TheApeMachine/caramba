from __future__ import annotations

import torch

from caramba.orchestrator.telemetry import SpikeDetector
from caramba.trainer.steppers.blockwise import _scale_lr


def test_scale_lr_updates_optimizer_groups() -> None:
    p = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.SGD([p], lr=1e-3)
    new_lr = _scale_lr(optimizer=opt, scheduler=None, factor=0.5, min_lr=1e-6)
    assert abs(new_lr - 5e-4) < 1e-12


def test_spike_detector_flags_large_jump() -> None:
    d = SpikeDetector(ema_decay=0.5, threshold_std=1.0, window_size=10)
    assert d.update(1.0) is False
    assert d.update(1.0) is False
    # Big jump should exceed ema + std
    assert d.update(10.0) is True

