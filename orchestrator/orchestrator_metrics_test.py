from __future__ import annotations

from torch import nn

from orchestrator.orchestrator import Orchestrator, OrchestratorConfig


def test_orchestrator_record_metrics_uses_config_keys() -> None:
    model = nn.Linear(3, 4)
    orch = Orchestrator(
        model=model,
        config=OrchestratorConfig(loss_key="generator_loss", grad_norm_key="gn", lr_key="lr"),
        portfolio=[],
    )
    snap = orch.record(metrics={"generator_loss": 1.25, "gn": 3.0, "lr": 1e-3, "disc_loss": 0.5})
    assert abs(snap.loss - 1.25) < 1e-9
    assert abs(snap.grad_norm - 3.0) < 1e-9
    assert abs(snap.lr - 1e-3) < 1e-12
    assert "disc_loss" in snap.metrics

