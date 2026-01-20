from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import caramba.trainer.initializers.default as init
from config.group import Group
from config.initializer import DefaultInitializerConfig
from config.model import ModelConfig
from config.train import TrainConfig, TrainPhase
from runtime.plan import RuntimePlan
from trainer.initializers.default import DefaultInitializer, _make_teacher_model_config
from trainer.upcycle_init_context import UpcycleInitContext


def test_make_teacher_model_config_rewrites_attention_to_standard() -> None:
    cfg = ModelConfig.model_validate(
        {
            "type": "TransformerModel",
            "topology": {
                "type": "StackedTopology",
                "layers": [
                    {
                        "type": "AttentionLayer",
                        "d_model": 32,
                        "n_heads": 4,
                        "mode": "decoupled",
                        "attn_dim": 16,
                        "sem_dim": 8,
                        "geo_dim": 8,
                        "decoupled_gate": True,
                        "decoupled_gate_dynamic": True,
                    }
                ],
            },
        }
    )
    t = _make_teacher_model_config(cfg)
    # The teacher config is validated into a full schema, so dropped keys may reappear
    # in a plain model_dump() as explicit defaults/None. When serialized (exclude None
    # and defaults), student-only knobs should be absent.
    d = t.model_dump()
    assert d["topology"]["layers"][0]["mode"] == "standard"

    d_ser = t.model_dump(exclude_none=True, exclude_defaults=True)
    attn = d_ser["topology"]["layers"][0]
    assert "attn_dim" not in attn
    assert "sem_dim" not in attn
    assert "geo_dim" not in attn
    assert "decoupled_gate" not in attn
    assert "decoupled_gate_dynamic" not in attn


def test_default_initializer_resolve_teacher_ckpt(tmp_path: Path) -> None:
    p_in = tmp_path / "x.pt"
    p = DefaultInitializer._resolve_teacher_ckpt(str(p_in))
    assert str(p).endswith(str(p_in))


def test_default_initializer_init_models_is_testable_with_monkeypatch(tmp_path: Path, monkeypatch) -> None:
    # Patch heavy components to minimal deterministic stubs.
    monkeypatch.setattr(init, "CheckpointBuilder", lambda: type("CB", (), {"load": lambda _self, _p: {"k": torch.tensor(1)}})())
    monkeypatch.setattr(
        init,
        "AdapterStateDictTransformer",
        type("A", (), {"llama": staticmethod(lambda **_kw: type("U", (), {"apply": staticmethod(lambda **_k: None)})())}),
    )

    class FakeModel(nn.Module):
        def __init__(self, cfg) -> None:
            super().__init__()
            self.cfg = cfg
            self.p = nn.Parameter(torch.tensor(1.0))

    monkeypatch.setattr(init, "Model", FakeModel)

    # Silence logging in tests.
    monkeypatch.setattr(init.logger, "header", lambda *_a, **_k: None)
    monkeypatch.setattr(init.logger, "info", lambda *_a, **_k: None)
    monkeypatch.setattr(init.logger, "success", lambda *_a, **_k: None)
    monkeypatch.setattr(init.logger, "warning", lambda *_a, **_k: None)

    plan = RuntimePlan(
        key="k",
        device="cpu",
        torch_version="x",
        dtype="float32",
        use_amp=False,
        amp_dtype="float16",
        batch_size=2,
        compile=False,
        compile_mode="reduce-overhead",
    )
    # Minimal manifest-like object with model payload.
    manifest = type(
        "M",
        (),
        {
            "model": {
                "type": "TransformerModel",
                "topology": {"type": "StackedTopology", "layers": [{"type": "LinearLayer", "d_in": 2, "d_out": 2, "bias": True}]},
            }
        },
    )()

    ctx = UpcycleInitContext(
        manifest=manifest,
        group=Group(name="test", description="", data="", runs=[]),
        defaults=None,
        checkpoint_dir=tmp_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
        runtime_plan=plan,
        dist_ctx=None,
    )

    train = TrainConfig(
        phase=TrainPhase.BLOCKWISE,
        batch_size=2,
        block_size=4,
        lr=1e-3,
        device="cpu",
        teacher_ckpt=str(tmp_path / "t.pt"),
        activation_checkpointing=True,
        activation_checkpoint_threshold_mb=12.0,
    )

    teacher, student = DefaultInitializer(DefaultInitializerConfig()).init_models(train, ctx)
    assert isinstance(teacher, nn.Module)
    assert isinstance(student, nn.Module)

