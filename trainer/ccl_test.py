from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from config.component import ComponentSpec
from config.defaults import Defaults, DefaultsData, DefaultsLogging
from config.manifest import Manifest
from config.mode import Mode
from config.run import Run
from config.target import ExperimentTargetConfig
from config.train import TrainConfig, TrainPhase
from trainer.ccl import CCLTrainer


class ToyStripeDataset(Dataset[dict[str, torch.Tensor]]):
    """Two-class toy images that are easy for patch-token models."""

    def __init__(self, n: int = 200, *, h: int = 8, w: int = 8, seed: int = 0) -> None:
        self.n = int(n)
        self.h = int(h)
        self.w = int(w)
        self.rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return int(self.n)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Alternate labels.
        y = int(idx % 2)
        img = np.zeros((1, self.h, self.w), dtype=np.float32)
        if y == 0:
            img[:, :, : self.w // 2] = 1.0
        else:
            img[:, :, self.w // 2 :] = 1.0
        img += 0.05 * self.rng.standard_normal(size=img.shape).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)
        return {"inputs": torch.from_numpy(img), "targets": torch.tensor(y, dtype=torch.long)}


@dataclass
class DatasetComponent:
    ds: Dataset[Any]

    def build(self) -> Dataset[Any]:
        return self.ds


class DummyRegistry:
    def __init__(self, train_comp: object) -> None:
        self._train_comp = train_comp

    def build(self, spec: ComponentSpec, *, backend: str) -> object:
        _ = backend
        if spec.ref == "dataset.toy":
            return self._train_comp
        raise KeyError(spec.ref)


class DummyEngine:
    def __init__(self, train_comp: object) -> None:
        self.registry = DummyRegistry(train_comp)


def test_ccl_trainer_runs_end_to_end_and_beats_chance(tmp_path: Path) -> None:
    ds_comp = DatasetComponent(ToyStripeDataset(n=200, h=8, w=8, seed=0))
    engine = DummyEngine(ds_comp)

    defaults = Defaults(
        data=DefaultsData(val_frac=0.2),
        logging=DefaultsLogging(wandb=False),
    )
    manifest = Manifest(version=2, defaults=defaults, targets=[])

    train = TrainConfig(phase=TrainPhase.STANDARD, batch_size=1, block_size=1, lr=1e-3, device="cpu")
    run = Run(id="fit", mode=Mode.TRAIN, exp="ccl_toy", seed=0, steps=1, expected={}, train=train)
    target = ExperimentTargetConfig(
        name="toy_ccl",
        backend="torch",
        task=ComponentSpec(ref="task.supervised_classification"),
        data=ComponentSpec(ref="dataset.toy"),
        system=ComponentSpec(ref="system.ccl", config={}),
        objective=ComponentSpec(ref="objective.none"),
        trainer=ComponentSpec(ref="trainer.ccl"),
        runs=[run],
    )

    tr = CCLTrainer(
        out_dir=str(tmp_path),
        k=32,
        patch=2,
        stride=2,
        alpha=0.5,
        seed=0,
        image_pool=64,
        sample_patches=4000,
        kmeans_max_iter=50,
        kmeans_batch_size=256,
        tokenize_batch_size=64,
        max_train=160,
        max_eval=40,
        generate=False,
    )

    out = tr.run(manifest=manifest, target=target, engine=engine, dry_run=False)
    assert isinstance(out, dict)
    assert "system" in out
    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert float(metrics["eval_accuracy"]) >= 0.7

