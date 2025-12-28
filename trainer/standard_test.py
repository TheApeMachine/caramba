from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from config.component import ComponentSpec
from config.defaults import Defaults, DefaultsLogging
from config.mode import Mode
from config.run import Run
from config.target import ExperimentTargetConfig
from config.train import TrainConfig, TrainPhase
from trainer.standard import StandardTrainer


class TinyDataset(Dataset):
    def __init__(self, n: int = 64) -> None:
        self.n = int(n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
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
    def loss(self, *, outputs, batch):
        y = outputs["y"]
        return (y.float().pow(2).mean())

    def metrics(self, *, outputs, batch, loss):
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
        trainer=ComponentSpec(ref="trainer.standard"),
        runs=[r],
    )
    return t, r


def test_parse_dtype_mapping() -> None:
    tr = StandardTrainer()
    assert tr._parse_dtype("float32") == torch.float32
    assert tr._parse_dtype("float16") == torch.float16
    assert tr._parse_dtype("bfloat16") == torch.bfloat16
    assert tr._parse_dtype("unknown") == torch.float32


def test_load_or_create_runtime_plan_reuses_cached_plan(tmp_path: Path) -> None:
    tr = StandardTrainer()
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
    plan1 = tr._load_or_create_runtime_plan(
        checkpoint_dir=tmp_path,
        device=torch.device("cpu"),
        train=train,
        system_cfg={"model": {"type": "TransformerModel", "topology": {"type": "StackedTopology", "layers": []}}},
    )
    plan2 = tr._load_or_create_runtime_plan(
        checkpoint_dir=tmp_path,
        device=torch.device("cpu"),
        train=train,
        system_cfg={"model": {"type": "TransformerModel", "topology": {"type": "StackedTopology", "layers": []}}},
    )
    assert plan1.key == plan2.key
    assert (tmp_path / "plans").exists()


def test_run_single_minimal_executes_training_loop_and_writes_checkpoint(tmp_path: Path) -> None:
    tr = StandardTrainer(checkpoint_dir=str(tmp_path))
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

    # Run directly to avoid registry wiring; this still covers almost all trainer logic.
    from instrumentation.run_logger import RunLogger

    rl = RunLogger(tmp_path, filename="train.jsonl", enabled=True)
    tr._run_single(
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

    # Check that checkpoint exists.
    ckpt = tmp_path / "r_standard_final.pt"
    assert ckpt.exists()


def test_run_single_unknown_optimizer_raises(tmp_path: Path) -> None:
    tr = StandardTrainer(checkpoint_dir=str(tmp_path))
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
    from instrumentation.run_logger import RunLogger

    rl = RunLogger(tmp_path, filename="train.jsonl", enabled=True)
    with pytest.raises(ValueError, match="Unknown optimizer"):
        tr._run_single(
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


def test_run_single_distributed_init_failure_is_wrapped(tmp_path: Path, monkeypatch) -> None:
    tr = StandardTrainer(checkpoint_dir=str(tmp_path))
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

    import trainer.distributed as dist

    monkeypatch.setattr(dist.DistributedContext, "init", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))

    from instrumentation.run_logger import RunLogger

    rl = RunLogger(tmp_path, filename="train.jsonl", enabled=True)
    with pytest.raises(RuntimeError, match="Failed to initialize distributed training"):
        tr._run_single(
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

