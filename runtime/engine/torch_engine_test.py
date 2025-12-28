from __future__ import annotations

import pytest
import torch
from torch import nn

from runtime.engine.torch_engine import TorchEngine
from config.component import ComponentSpec
from config.mode import Mode
from config.run import Run
from config.target import ExperimentTargetConfig
from config.train import TrainConfig, TrainPhase


class Wrapper:
    def __init__(self, module: nn.Module) -> None:
        self.module = module


def test_engine_registers_builtin_components_and_can_build_some() -> None:
    e = TorchEngine()
    # Should be able to build a built-in trainer and objective via the registry.
    trainer = e.registry.build(ComponentSpec(ref="trainer.standard"), backend="torch")
    assert hasattr(trainer, "run")

    obj = e.registry.build(ComponentSpec(ref="objective.mse"), backend="torch")
    assert hasattr(obj, "loss")


def test_as_module_accepts_module_and_wrapper_and_rejects_other() -> None:
    m = nn.Linear(2, 2)
    assert TorchEngine._as_module(m) is m
    assert TorchEngine._as_module(Wrapper(m)) is m
    with pytest.raises(TypeError):
        TorchEngine._as_module(object())


def test_first_train_returns_first_run_with_train_or_object() -> None:
    tcfg = TrainConfig(phase=TrainPhase.STANDARD, batch_size=1, block_size=4, lr=1e-3, device="cpu")
    r1 = Run(id="a", mode=Mode.TRAIN, exp="e", seed=0, steps=1, expected={}, train=None)
    r2 = Run(id="b", mode=Mode.TRAIN, exp="e", seed=0, steps=1, expected={}, train=tcfg)
    target = ExperimentTargetConfig(
        name="exp",
        backend="torch",
        task=ComponentSpec(ref="task.language_modeling"),
        data=ComponentSpec(ref="dataset.tokens", config={"path": "x.tokens", "block_size": 4}),
        system=ComponentSpec(ref="system.generic", config={"model": {"type": "TransformerModel", "topology": {"type": "StackedTopology", "layers": []}}}),
        objective=ComponentSpec(ref="objective.mse"),
        trainer=ComponentSpec(ref="trainer.standard"),
        runs=[r1, r2],
    )
    got = TorchEngine._first_train(target)
    assert got is tcfg

