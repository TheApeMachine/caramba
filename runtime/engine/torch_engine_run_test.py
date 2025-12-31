from __future__ import annotations

import pytest
import torch
from torch import nn

from caramba.runtime.engine.torch_engine import TorchEngine
from caramba.config.component import ComponentSpec
from caramba.config.defaults import Defaults
from caramba.config.manifest import Manifest
from caramba.config.mode import Mode
from caramba.config.run import Run
from caramba.config.target import ExperimentTargetConfig
from caramba.config.train import TrainConfig, TrainPhase


def test_torch_engine_run_experiment_dry_run_short_circuits() -> None:
    e = TorchEngine()
    m = Manifest(version=2, defaults=Defaults(), targets=[])
    t = ExperimentTargetConfig(
        name="exp",
        backend="torch",
        task=ComponentSpec(ref="task.language_modeling"),
        data=ComponentSpec(ref="dataset.tokens", config={"path": "x.tokens", "block_size": 4}),
        system=ComponentSpec(ref="system.generic", config={"model": {"type": "TransformerModel", "topology": {"type": "StackedTopology", "layers": []}}}),
        objective=ComponentSpec(ref="objective.mse"),
        trainer=ComponentSpec(ref="trainer.standard"),
        runs=[],
    )
    r = e.run_experiment(m, t, dry_run=True)
    assert r is None


def test_torch_engine_run_experiment_runs_metrics_best_effort(monkeypatch) -> None:
    e = TorchEngine()

    class DummyTrainer:
        def run(self, *, manifest, target, engine, dry_run=False):
            return {"system": nn.Linear(2, 2), "device": torch.device("cpu")}

    class GoodMetric:
        def run(self, *, model, device, name="model"):
            return {"ok": 1.0}

    class BadMetric:
        def run(self, *, model, device, name="model"):
            raise RuntimeError("boom")

    # Force trainer build to our dummy, and metrics build per-ref.
    def fake_build(spec, *, backend):
        if spec.ref == "trainer.standard":
            return DummyTrainer()
        if spec.ref == "metric.good":
            return GoodMetric()
        if spec.ref == "metric.bad":
            return BadMetric()
        raise KeyError(spec.ref)

    monkeypatch.setattr(e.registry, "build", fake_build)

    train = TrainConfig(phase=TrainPhase.STANDARD, batch_size=1, block_size=4, lr=1e-3, device="cpu")
    run = Run(id="r", mode=Mode.TRAIN, exp="e", seed=0, steps=1, expected={}, train=train)
    t = ExperimentTargetConfig(
        name="exp",
        backend="torch",
        task=ComponentSpec(ref="task.language_modeling"),
        data=ComponentSpec(ref="dataset.tokens", config={"path": "x.tokens", "block_size": 4}),
        system=ComponentSpec(ref="system.generic", config={"model": {"type": "TransformerModel", "topology": {"type": "StackedTopology", "layers": []}}}),
        objective=ComponentSpec(ref="objective.mse"),
        trainer=ComponentSpec(ref="trainer.standard"),
        runs=[run],
        metrics=[ComponentSpec(ref="metric.good"), ComponentSpec(ref="metric.bad"), ComponentSpec(ref="metric.missing")],
    )
    m = Manifest(version=2, defaults=Defaults(), targets=[t])

    # Should not raise even when metrics fail to build or run.
    out = e.run_experiment(m, t, dry_run=False)
    assert isinstance(out, dict)

