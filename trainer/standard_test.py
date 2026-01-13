from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from caramba.config.component import ComponentSpec
from caramba.config.defaults import Defaults, DefaultsLogging
from caramba.config.mode import Mode
from caramba.config.run import Run
from caramba.config.target import ExperimentTargetConfig
from caramba.config.train import TrainConfig, TrainPhase
from caramba.trainer.standard_stepper import StandardTrainStepper
from caramba.trainer.stepper import TrainSession


class TinyDataset(Dataset):
    def __init__(self, n: int = 64) -> None:
        self.n = int(n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Return dict to exercise as_tensordict/collate_tensordict path.
        x = torch.tensor([float(idx), float(idx + 1)], dtype=torch.float32)
        return {"x": x}


@dataclass
class DatasetComponent:
    ds: Dataset

    def build(self):
        return self.ds


class DummySystem(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.1))

    def forward(self, batch):  # TensorDict-like / dict-like
        x = batch["x"].float()
        # Produce a tensor output and a dict wrapper.
        y = x.sum(dim=-1, keepdim=True) + self.w
        return {"y": y}


class DummyObjective:
    def loss(self, *, outputs, _batch):
        y = outputs["y"]
        return (y.float().pow(2).mean())

    def metrics(self, *, outputs, _batch, _loss):
        return {"y_mean": float(outputs["y"].detach().mean())}


def _target_and_run(*, steps: int, train: TrainConfig) -> tuple[ExperimentTargetConfig, Run]:
    r = Run(id="r", mode=Mode.TRAIN, exp="e", seed=0, steps=int(steps), expected={}, train=train)
    t = ExperimentTargetConfig(
        name="exp",
        backend="torch",
        task=ComponentSpec(ref="task.language_modeling"),
        data=ComponentSpec(ref="dataset.tokens", config={"path": "x.tokens", "block_size": 4}),
        system=ComponentSpec(ref="system.generic", config={"model": {"type": "TransformerModel", "topology": {"type": "StackedTopology", "layers": [{"type": "LinearLayer", "d_in": 2, "d_out": 2, "bias": True}]}}}),
        objective=ComponentSpec(ref="objective.mse"),
        trainer=ComponentSpec(ref="trainer.train"),
        runs=[r],
    )
    return t, r


def test_parse_dtype_mapping() -> None:
    stepper = StandardTrainStepper()
    assert stepper._parse_dtype("float32") == torch.float32
    assert stepper._parse_dtype("float16") == torch.float16
    assert stepper._parse_dtype("bfloat16") == torch.bfloat16
    assert stepper._parse_dtype("unknown") == torch.float32


def test_load_or_create_runtime_plan_reuses_cached_plan(tmp_path: Path) -> None:
    stepper = StandardTrainStepper()
    train = TrainConfig(
        phase=TrainPhase.STANDARD,
        batch_size=8,
        block_size=16,
        lr=1e-3,
        device="cpu",
        dtype="auto",
        auto_batch_size=True,
        auto_batch_ref_block_size=512,
        auto_batch_min=2,
    )
    plan1 = stepper._load_or_create_runtime_plan(
        checkpoint_dir=tmp_path,
        device=torch.device("cpu"),
        train=train,
        system_cfg={"model": {"type": "TransformerModel", "topology": {"type": "StackedTopology", "layers": []}}},
    )
    plan2 = stepper._load_or_create_runtime_plan(
        checkpoint_dir=tmp_path,
        device=torch.device("cpu"),
        train=train,
        system_cfg={"model": {"type": "TransformerModel", "topology": {"type": "StackedTopology", "layers": []}}},
    )
    assert plan1.key == plan2.key
    assert (tmp_path / "plans").exists()


def test_run_single_minimal_executes_training_loop_and_writes_checkpoint(tmp_path: Path) -> None:
    stepper = StandardTrainStepper()
    ds_comp = DatasetComponent(TinyDataset(64))
    system = DummySystem()
    objective = DummyObjective()
    defaults = Defaults(logging=DefaultsLogging(wandb=False))

    train = TrainConfig(
        phase=TrainPhase.STANDARD,
        batch_size=4,
        block_size=4,
        lr=1e-2,
        device="cpu",
        telemetry_interval=1,
        optimizer="adamw",
        offload_optimizer=True,
        use_amp=False,
    )
    target, run = _target_and_run(steps=2, train=train)

    from caramba.instrumentation.run_logger import RunLogger

    with RunLogger(tmp_path, filename="train.jsonl", enabled=True) as rl:
        stepper.run(
            TrainSession(
                defaults=defaults,
                target=target,
                run=run,
                train=train,
                dataset_comp=ds_comp,
                system=system,
                objective=objective,
                checkpoint_dir=tmp_path,
                run_logger=rl,
            )
        )

    # Check that checkpoint exists.
    ckpt = tmp_path / "r_standard_final.pt"
    assert ckpt.exists()


def test_run_single_unknown_optimizer_raises(tmp_path: Path) -> None:
    stepper = StandardTrainStepper()
    ds_comp = DatasetComponent(TinyDataset(32))
    system = DummySystem()
    objective = DummyObjective()
    defaults = Defaults(logging=DefaultsLogging(wandb=False))

    train = TrainConfig(
        phase=TrainPhase.STANDARD,
        batch_size=4,
        block_size=4,
        lr=1e-2,
        device="cpu",
        optimizer="nope",
    )
    target, run = _target_and_run(steps=1, train=train)
    from caramba.instrumentation.run_logger import RunLogger

    with pytest.raises(ValueError, match="Unknown optimizer"):
        with RunLogger(tmp_path, filename="train.jsonl", enabled=True) as rl:
            stepper.run(
                TrainSession(
                    defaults=defaults,
                    target=target,
                    run=run,
                    train=train,
                    dataset_comp=ds_comp,
                    system=system,
                    objective=objective,
                    checkpoint_dir=tmp_path,
                    run_logger=rl,
                )
            )


def test_run_single_distributed_init_failure_is_wrapped(tmp_path: Path, monkeypatch) -> None:
    stepper = StandardTrainStepper()
    ds_comp = DatasetComponent(TinyDataset(32))
    system = DummySystem()
    objective = DummyObjective()
    defaults = Defaults(logging=DefaultsLogging(wandb=False))

    train = TrainConfig(
        phase=TrainPhase.STANDARD,
        batch_size=4,
        block_size=4,
        lr=1e-2,
        device="cpu",
        distributed_strategy="ddp",
        distributed_backend="gloo",
    )
    target, run = _target_and_run(steps=1, train=train)

    import caramba.trainer.distributed as dist

    def raise_boom(*_a, **_k) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(dist.DistributedContext, "init", raise_boom)

    from caramba.instrumentation.run_logger import RunLogger

    with pytest.raises(RuntimeError, match="Failed to initialize distributed training"):
        with RunLogger(tmp_path, filename="train.jsonl", enabled=True) as rl:
            stepper.run(
                TrainSession(
                    defaults=defaults,
                    target=target,
                    run=run,
                    train=train,
                    dataset_comp=ds_comp,
                    system=system,
                    objective=objective,
                    checkpoint_dir=tmp_path,
                    run_logger=rl,
                )
            )
