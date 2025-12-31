from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pytest
import torch
from torch.utils.data import Dataset

import caramba.trainer.collectors.default as dc
from caramba.config.collector import DefaultCollectorConfig
from caramba.config.defaults import Defaults, DefaultsData
from caramba.config.group import Group
from caramba.config.mode import Mode
from caramba.config.run import Run
from caramba.config.train import TrainConfig, TrainPhase
from caramba.runtime.plan import RuntimePlan
from caramba.trainer.collectors.default import DefaultCollector, _resolve_data_path
from caramba.trainer.upcycle_context import UpcycleContext


class TinyTokenDataset(Dataset):
    def __init__(self, n: int = 32) -> None:
        self.n = int(n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"token_ids": torch.tensor([idx, idx + 1], dtype=torch.long)}


def test_resolve_data_path_prefers_existing_absolute(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_text("x", encoding="utf-8")
    assert _resolve_data_path(str(p)) == p


def test_default_collector_build_loaders_uses_monkeypatched_dataset(tmp_path: Path, monkeypatch) -> None:
    # Make the data path "exist" without needing a real token file.
    f = tmp_path / "x.tokens"
    f.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(dc, "_resolve_data_path", lambda _spec: f)
    monkeypatch.setattr(dc, "build_token_dataset", lambda *, path, block_size: TinyTokenDataset(40))

    train = TrainConfig(phase=TrainPhase.STANDARD, batch_size=4, block_size=4, lr=1e-3, device="cpu")
    run = Run(id="r", mode=Mode.TRAIN, exp="e", seed=0, steps=1, expected={}, train=train)
    group = Group(name="g", description="", data=str(f), runs=[run])

    plan = RuntimePlan(
        key="k",
        device="cpu",
        torch_version="x",
        dtype="float32",
        use_amp=False,
        amp_dtype="float16",
        batch_size=4,
        compile=False,
        compile_mode="reduce-overhead",
    )
    defaults = Defaults(data=DefaultsData(val_frac=0.1))
    ctx = UpcycleContext(
        manifest=object(),
        group=group,
        defaults=defaults,
        checkpoint_dir=tmp_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
        runtime_plan=plan,
        teacher=torch.nn.Linear(2, 2),
        student=torch.nn.Linear(2, 2),
        inst=None,
        dist_ctx=None,
    )

    c = DefaultCollector(DefaultCollectorConfig())
    train_loader, val_loader = c.build_loaders(train, ctx)
    b = next(iter(train_loader))
    assert "token_ids" in dict(b)
    assert val_loader is not None


def test_default_collector_next_batch_resets_on_stop_iteration() -> None:
    # Minimal loader/iterator pair for next_batch.
    ds = TinyTokenDataset(8)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, drop_last=True, collate_fn=dc.collate_tensordict)
    it = iter(loader)
    c = DefaultCollector(DefaultCollectorConfig())
    _b1, it = c.next_batch(loader, it)
    # Exhaust iterator.
    list(it)
    b2, _it2 = c.next_batch(loader, iter([]))  # force StopIteration path immediately
    assert "token_ids" in dict(b2)

