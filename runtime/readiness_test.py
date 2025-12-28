from __future__ import annotations

import importlib.util

import pytest

import runtime.readiness as readiness
from config.benchmark import BenchmarkSpec, LatencyBenchmarkConfig, MemoryBenchmarkConfig
from config.component import ComponentSpec
from config.defaults import Defaults
from config.manifest import Manifest
from config.mode import Mode
from config.run import Run
from config.target import ExperimentTargetConfig
from config.train import TrainConfig, TrainPhase


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
        trainer=ComponentSpec(ref="trainer.standard"),
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
        trainer=ComponentSpec(ref="trainer.standard"),
        runs=[run],
        benchmarks=None,
    )


def test_plotting_requested_warns_when_matplotlib_missing(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    m = _manifest_with_instrument("rich,liveplot")
    t = _non_decoupled_target(device="cpu")

    rep = readiness.check_target_readiness(m, t)
    assert rep.errors == []
    assert any(i.code == "matplotlib_missing" for i in rep.warnings)


def test_no_decoupled_attention_skips_perf_backend_checks(monkeypatch) -> None:
    # Even if perf backends are "missing", without DBA attention we don't care.
    monkeypatch.setattr(readiness, "METAL_SUPPORTED", False)
    monkeypatch.setattr(readiness, "TRITON_AVAILABLE", False)

    m = Manifest.model_validate({"version": 2, "defaults": Defaults().model_dump(), "targets": []})
    t = _non_decoupled_target(device="mps")
    rep = readiness.check_target_readiness(m, t, best_effort=False)
    assert rep.errors == []


def test_mps_unavailable_is_error_when_decoupled_and_fp16_cache(monkeypatch) -> None:
    monkeypatch.setattr(readiness, "METAL_SUPPORTED", False)
    monkeypatch.setattr(readiness, "METAL_BUILD_TOOLS_AVAILABLE", True)

    b = BenchmarkSpec(id="lat", config=LatencyBenchmarkConfig(cache_kind="fp16"))
    t = _decoupled_target(device="mps", benchmarks=[b])
    m = Manifest.model_validate({"version": 2, "defaults": Defaults().model_dump(), "targets": []})

    rep = readiness.check_target_readiness(m, t, best_effort=False)
    assert any(i.code == "mps_unavailable" for i in rep.errors)


def test_metal_build_tools_missing_is_error_or_warning_based_on_best_effort(monkeypatch) -> None:
    monkeypatch.setattr(readiness, "METAL_SUPPORTED", True)
    monkeypatch.setattr(readiness, "METAL_BUILD_TOOLS_AVAILABLE", False)

    b = BenchmarkSpec(id="lat", config=LatencyBenchmarkConfig(cache_kind="auto"))
    t = _decoupled_target(device="mps", benchmarks=[b])
    m = Manifest.model_validate({"version": 2, "defaults": Defaults().model_dump(), "targets": []})

    rep_err = readiness.check_target_readiness(m, t, best_effort=False)
    assert any(i.code == "metal_build_tools_missing" for i in rep_err.errors)

    rep_warn = readiness.check_target_readiness(m, t, best_effort=True)
    assert any(i.code == "metal_build_tools_missing" for i in rep_warn.warnings)


def test_triton_missing_for_cuda_q4_is_error_or_warning(monkeypatch) -> None:
    monkeypatch.setattr(readiness, "TRITON_AVAILABLE", False)

    b = BenchmarkSpec(id="mem", config=MemoryBenchmarkConfig(quantization_modes=["q4_0"]))
    t = _decoupled_target(device="cuda", benchmarks=[b])
    m = Manifest.model_validate({"version": 2, "defaults": Defaults().model_dump(), "targets": []})

    rep_err = readiness.check_target_readiness(m, t, best_effort=False)
    assert any(i.code == "triton_missing" for i in rep_err.errors)

    rep_warn = readiness.check_target_readiness(m, t, best_effort=True)
    assert any(i.code == "triton_missing" for i in rep_warn.warnings)


def test_format_readiness_report_formats_both_errors_and_warnings() -> None:
    rep = readiness.ReadinessReport(
        errors=[readiness.ReadinessIssue(kind="error", code="e", message="bad")],
        warnings=[readiness.ReadinessIssue(kind="warning", code="w", message="meh")],
    )
    txt = readiness.format_readiness_report(rep)
    assert "ERROR [e] bad" in txt
    assert "WARN [w] meh" in txt

