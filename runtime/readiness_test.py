from __future__ import annotations

import importlib.util

import pytest

import caramba.runtime.readiness as readiness
from caramba.config.benchmark import BenchmarkSpec, LatencyBenchmarkConfig, MemoryBenchmarkConfig
from caramba.config.component import ComponentSpec
from caramba.config.defaults import Defaults
from caramba.config.manifest import Manifest
from caramba.config.mode import Mode
from caramba.config.run import Run
from caramba.config.target import ExperimentTargetConfig
from caramba.config.train import TrainConfig, TrainPhase


def _default_manifest() -> Manifest:
    return Manifest.model_validate(
        {"version": 2, "defaults": Defaults().model_dump(), "targets": []}
    )


def _manifest_with_instrument(instrument: str) -> Manifest:
    # Build via model_validate to keep it close to real manifests.
    return Manifest.model_validate(
        {
            "version": 2,
            "name": "t",
            "defaults": {"logging": {"instrument": instrument}},
            "targets": [],
        }
    )


def _decoupled_target(*, device: str, benchmarks: list[BenchmarkSpec] | None) -> ExperimentTargetConfig:
    train = TrainConfig(
        phase=TrainPhase.STANDARD,
        batch_size=2,
        block_size=4,
        lr=1e-3,
        device=device,
    )
    run = Run(id="r", mode=Mode.TRAIN, exp="e", seed=0, steps=1, expected={}, train=train)

    # Minimal model with one decoupled attention layer.
    model_payload = {
        "type": "TransformerModel",
        "topology": {
            "type": "StackedTopology",
            "layers": [
                {
                    "type": "AttentionLayer",
                    "d_model": 32,
                    "n_heads": 4,
                    "mode": "decoupled",
                    "sem_dim": 16,
                    "geo_dim": 16,
                }
            ],
        },
    }

    return ExperimentTargetConfig(
        name="exp",
        backend="torch",
        task=ComponentSpec(ref="task.language_modeling"),
        data=ComponentSpec(ref="dataset.tokens", config={"path": "x.tokens", "block_size": 4}),
        system=ComponentSpec(ref="system.language_model", config={"model": model_payload}),
        objective=ComponentSpec(ref="objective.next_token_ce"),
        trainer=ComponentSpec(ref="trainer.train"),
        runs=[run],
        benchmarks=benchmarks,
    )


def _non_decoupled_target(*, device: str) -> ExperimentTargetConfig:
    train = TrainConfig(
        phase=TrainPhase.STANDARD,
        batch_size=2,
        block_size=4,
        lr=1e-3,
        device=device,
    )
    run = Run(id="r", mode=Mode.TRAIN, exp="e", seed=0, steps=1, expected={}, train=train)
    model_payload = {
        "type": "TransformerModel",
        "topology": {
            "type": "StackedTopology",
            "layers": [{"type": "LinearLayer", "d_in": 8, "d_out": 8, "bias": True}],
        },
    }
    return ExperimentTargetConfig(
        name="exp",
        backend="torch",
        task=ComponentSpec(ref="task.language_modeling"),
        data=ComponentSpec(ref="dataset.tokens", config={"path": "x.tokens", "block_size": 4}),
        system=ComponentSpec(ref="system.language_model", config={"model": model_payload}),
        objective=ComponentSpec(ref="objective.next_token_ce"),
        trainer=ComponentSpec(ref="trainer.train"),
        runs=[run],
        benchmarks=None,
    )


def test_plotting_requested_warns_when_matplotlib_missing(monkeypatch) -> None:
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None):
        if name == "matplotlib" or name.startswith("matplotlib."):
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    m = _manifest_with_instrument("rich,liveplot")
    t = _non_decoupled_target(device="cpu")

    rep = readiness.check_target_readiness(m, t)
    assert rep.errors == []
    assert any(i.code == "matplotlib_missing" for i in rep.warnings)


def test_no_decoupled_attention_skips_perf_backend_checks(monkeypatch) -> None:
    # Even if perf backends are "missing", without DBA attention we don't care.
    monkeypatch.setattr(readiness, "metal_supported", False)
    monkeypatch.setattr(readiness, "triton_supported", False)

    m = _default_manifest()
    t = _non_decoupled_target(device="mps")
    rep = readiness.check_target_readiness(m, t)
    assert rep.errors == []


def test_mps_unavailable_is_error_when_decoupled_and_fp16_cache(monkeypatch) -> None:
    monkeypatch.setattr(readiness, "metal_supported", False)
    monkeypatch.setattr(readiness, "metal_build_tools_available", True)

    b = BenchmarkSpec(id="lat", config=LatencyBenchmarkConfig(cache_kind="fp16"))
    t = _decoupled_target(device="mps", benchmarks=[b])
    m = _default_manifest()

    rep = readiness.check_target_readiness(m, t)
    assert any(i.code == "mps_unavailable" for i in rep.errors)


def test_metal_build_tools_missing_is_error(monkeypatch) -> None:
    monkeypatch.setattr(readiness, "metal_supported", True)
    monkeypatch.setattr(readiness, "metal_build_tools_available", False)

    b = BenchmarkSpec(id="lat", config=LatencyBenchmarkConfig(cache_kind="auto"))
    t = _decoupled_target(device="mps", benchmarks=[b])
    m = _default_manifest()

    rep = readiness.check_target_readiness(m, t)
    assert any(i.code == "metal_build_tools_missing" for i in rep.errors)


def test_triton_missing_for_cuda_q4_is_error(monkeypatch) -> None:
    monkeypatch.setattr(readiness, "triton_supported", False)

    b = BenchmarkSpec(id="mem", config=MemoryBenchmarkConfig(quantization_modes=["q4_0"]))
    t = _decoupled_target(device="cuda", benchmarks=[b])
    m = _default_manifest()

    rep = readiness.check_target_readiness(m, t)
    assert any(i.code == "triton_missing" for i in rep.errors)


def test_format_readiness_report_formats_both_errors_and_warnings() -> None:
    rep = readiness.ReadinessReport(
        errors=[readiness.ReadinessIssue(kind="error", code="e", message="bad")],
        warnings=[readiness.ReadinessIssue(kind="warning", code="w", message="meh")],
    )
    txt = readiness.format_readiness_report(rep)
    assert "ERROR [e] bad" in txt
    assert "WARN [w] meh" in txt
