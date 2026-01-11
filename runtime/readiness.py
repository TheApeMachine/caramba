"""Runtime readiness checks (no silent fallbacks).

This module evaluates a manifest target's requested capabilities (device, cache
quantization, plotting) against what the current environment can provide.

Policy:
- Missing *performance* backends (Metal/Triton) are **errors** by default.
- Non-critical visualization deps (matplotlib) are **warnings**.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ValidationError

from caramba.config.benchmark import LatencyBenchmarkConfig, MemoryBenchmarkConfig
from caramba.config.layer import AttentionLayerConfig, AttentionMode
from caramba.config.manifest import Manifest
from caramba.config.model import ModelConfig
from caramba.config.target import ExperimentTargetConfig
from caramba.optimizer.runtime import (
    metal_build_tools_available,
    metal_supported,
    triton_supported,
)


@dataclass(frozen=True, slots=True)
class ReadinessIssue:
    kind: Literal["error", "warning"]
    code: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    errors: list[ReadinessIssue] = field(default_factory=list)
    warnings: list[ReadinessIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _has_decoupled_attention(model: ModelConfig) -> bool:
    def _walk(node: object) -> Iterable[object]:
        layers = getattr(node, "layers", None)
        if isinstance(layers, list):
            repeat = int(getattr(node, "repeat", 1) or 1)
            repeat = max(1, repeat)
            for _ in range(repeat):
                for child in layers:
                    yield from _walk(child)
            return
        yield node

    for n in _walk(model.topology):
        if isinstance(n, AttentionLayerConfig) and n.mode == AttentionMode.DECOUPLED:
            return True
    return False


def _collect_models(target: ExperimentTargetConfig) -> list[ModelConfig]:
    if target.system.ref not in ("system.language_model", "system.generic"):
        return []
    model_payload = target.system.config.get("model", None)
    if not isinstance(model_payload, dict):
        return []
    try:
        return [ModelConfig.model_validate(model_payload)]
    except ValidationError:
        return []


def _devices_used(target: ExperimentTargetConfig) -> set[str]:
    devices: set[str] = set()
    for r in target.runs:
        if r.train is None:
            continue
        devices.add(str(r.train.device).lower())
    if not devices:
        # Some experiment targets are benchmark-only (checkpoint reuse) and may not
        # include runs. In that case, honor an explicit trainer config device if present.
        try:
            dev = target.trainer.config.get("device", None)  # type: ignore[union-attr]
        except Exception:
            dev = None
        if isinstance(dev, str) and dev.strip():
            devices.add(dev.strip().lower())
        else:
            devices.add("cpu")
    return devices


def _cache_kinds_requested(target: ExperimentTargetConfig) -> set[str]:
    kinds: set[str] = set()
    if not target.benchmarks:
        return kinds
    for spec in target.benchmarks:
        cfg = spec.config
        if isinstance(cfg, LatencyBenchmarkConfig):
            if bool(getattr(cfg, "use_cache", True)):
                kinds.add(str(getattr(cfg, "cache_kind", "auto")).lower())
                pol = getattr(cfg, "cache_policy", None)
                if pol is not None:
                    # Collect all explicit tensor kinds referenced by the policy.
                    for k in ("k", "k_sem", "k_geo", "v"):
                        t = getattr(pol, k, None)
                        if t is not None:
                            kind = getattr(t, "kind", None)
                            if kind is not None:
                                kinds.add(str(kind).lower())
        if isinstance(cfg, MemoryBenchmarkConfig):
            for q in getattr(cfg, "quantization_modes", []) or []:
                kinds.add(str(q).lower())
    return kinds


def _plotting_requested(manifest: Manifest) -> bool:
    instrument = str(getattr(getattr(manifest.defaults, "logging", object()), "instrument", ""))
    tokens = {t for t in re.split(r"[,+\s]+", instrument.lower()) if t}
    return bool({"plot", "live", "liveplot"} & tokens)


def _matplotlib_available() -> bool:
    try:
        return importlib.util.find_spec("matplotlib") is not None
    except Exception:
        return False


def _probe_flag(value: object) -> bool:
    """Interpret a readiness probe that may be a bool or a 0-arg callable.

    Tests monkeypatch these probes to booleans; production code provides callables.
    """
    if callable(value):
        try:
            return bool(value())
        except TypeError:
            # If a non-0-arg callable slips through, treat it as truthy/falsey.
            return bool(value)
    return bool(value)


def check_target_readiness(
    manifest: Manifest,
    target: ExperimentTargetConfig,
) -> ReadinessReport:
    """Compute readiness issues for an experiment target."""
    report = ReadinessReport()

    models = _collect_models(target)
    has_decoupled = any(_has_decoupled_attention(m) for m in models)

    devices = _devices_used(target)
    cache_kinds = _cache_kinds_requested(target)

    # Visualization deps (warn-only).
    if _plotting_requested(manifest) and not _matplotlib_available():
        report.warnings.append(
            ReadinessIssue(
                kind="warning",
                code="matplotlib_missing",
                message="matplotlib is not available; live plotting will be disabled.",
            )
        )

    # Performance backends: only relevant if decoupled attention is present and
    # the target requests caches/quantization that activate the fused paths.
    if not has_decoupled:
        return report

    # Metal (MPS) fused decode is used for fp16 caches on MPS.
    if "mps" in devices and ("fp16" in cache_kinds or "auto" in cache_kinds):
        if not _probe_flag(metal_supported):
            report.errors.append(
                ReadinessIssue(
                    kind="error",
                    code="mps_unavailable",
                    message="Target requests device=mps, but PyTorch MPS is not available.",
                )
            )
        elif not _probe_flag(metal_build_tools_available):
            report.errors.append(
                ReadinessIssue(
                    kind="error",
                code="metal_build_tools_missing",
                message=(
                    "Metal build tools are not available (missing `metal`/`metallib` in the active Xcode toolchain); "
                    "fused DBA decode cannot be built and the run would fall back to unoptimized PyTorch.\n"
                    "Install/select Xcode Command Line Tools:\n"
                    "  - `xcode-select --install`\n"
                    "Or, if you have Xcode.app:\n"
                    "  - `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`\n"
                    "  - `sudo xcodebuild -license accept`\n"
                    "Verify:\n"
                    "  - `xcrun -sdk macosx --find metal`\n"
                    "  - `xcrun -sdk macosx --find metallib`"
                ),
                )
            )

    # Triton fused decode is used for quantized caches on CUDA.
    wants_q4 = any(k.startswith("q4") or k == "q4_0" for k in cache_kinds)
    if "cuda" in devices and wants_q4:
        if not _probe_flag(triton_supported):
            report.errors.append(
                ReadinessIssue(
                    kind="error",
                    code="triton_missing",
                    message=(
                        "Triton is not available; fused decoupled decode for quantized KV caches cannot run "
                        "and the run would fall back to unoptimized PyTorch."
                    ),
                )
            )

    return report


def format_readiness_report(report: ReadinessReport) -> str:
    lines: list[str] = []
    for issue in report.errors + report.warnings:
        prefix = "ERROR" if issue.kind == "error" else "WARN"
        lines.append(f"{prefix} [{issue.code}] {issue.message}")
    return "\n".join(lines).strip()

