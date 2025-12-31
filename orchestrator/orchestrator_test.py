from __future__ import annotations

import math

import torch
from torch import nn

from caramba.orchestrator.orchestrator import DecisionBoundary, Orchestrator, OrchestratorConfig
from caramba.orchestrator.telemetry import TelemetrySnapshot


def test_orchestrator_safety_baseline_triggers() -> None:
    model = nn.Linear(4, 4)
    orch = Orchestrator(model=model, config=OrchestratorConfig(max_loss_increase=1.5))
    orch.set_loss_baseline(10.0)

    snap = TelemetrySnapshot(loss=16.0)
    assert orch.should_evaluate(step=1, snapshot=snap) == DecisionBoundary.SAFETY


def test_orchestrator_safety_nan_triggers() -> None:
    model = nn.Linear(4, 4)
    orch = Orchestrator(model=model, config=OrchestratorConfig())

    snap = TelemetrySnapshot(loss=float("nan"))
    assert orch.should_evaluate(step=1, snapshot=snap) == DecisionBoundary.SAFETY

    snap2 = TelemetrySnapshot(loss=float("inf"))
    assert orch.should_evaluate(step=1, snapshot=snap2) == DecisionBoundary.SAFETY

    assert math.isfinite(orch.record(loss=1.0, grad_norm=0.0, lr=1e-4).loss)

