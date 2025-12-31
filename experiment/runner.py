"""Experiment orchestration (manifest v2).

Manifest v2 is target-based:
- targets are runnable units (experiments or agent processes)
- experiments are executed via an Engine (backend) which resolves components
  through the registry.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from caramba.compiler import Compiler
from caramba.config.manifest import Manifest
from caramba.config.target import ExperimentTargetConfig, ProcessTargetConfig, TargetConfig
from caramba.console import logger

from caramba.runtime.engine import TorchEngine
from caramba.runtime.readiness import check_target_readiness, format_readiness_report


def _resolve_target(manifest: Manifest, target: str | None) -> str:
    """Resolve a target name for execution."""
    if target:
        if manifest.entrypoints and target in manifest.entrypoints:
            target = manifest.entrypoints[target]
        if ":" in target:
            kind, name = target.split(":", 1)
            kind = kind.strip().lower()
            if kind in {"target", "experiment", "process"}:
                return name.strip()
        return target.strip()

    if manifest.entrypoints and "default" in manifest.entrypoints:
        return _resolve_target(manifest, manifest.entrypoints["default"])

    if manifest.targets:
        return manifest.targets[0].name

    raise ValueError("Manifest has no runnable targets.")


def _parse_target(target: str) -> tuple[str, str]:
    if ":" not in target:
        raise ValueError(
            f"Invalid target '{target}'. Expected 'target:<name>' or a bare target name."
        )
    kind, name = target.split(":", 1)
    kind = kind.strip().lower()
    name = name.strip()
    if kind not in {"target", "experiment", "process"}:
        raise ValueError(
            f"Invalid target kind '{kind}' for '{target}'. Expected 'target', 'experiment', or 'process'."
        )
    if not name:
        raise ValueError(f"Invalid target '{target}': missing name after ':'.")
    return kind, name


def run_from_manifest_path(
    manifest_path: Path,
    *,
    target: str | None = None,
    dry_run: bool = False,
) -> Any:
    """Single manifest-driven entrypoint for the whole platform."""
    manifest = Manifest.from_path(manifest_path)
    # If no target is specified, "run the manifest" means run every target in order.
    # The CLI should stay out of the way: no default entrypoints, no first-target guess.
    if target is None:
        if dry_run:
            return {"targets": [t.name for t in manifest.targets]}
        compiler = Compiler()
        lowered = compiler.lowerer.lower_manifest(manifest)
        compiler.validator.validate_manifest(lowered, print_plan=True)
        return ExperimentRunner(lowered).run_all(manifest_path=manifest_path)

    resolved = _resolve_target(manifest, target)
    name = resolved

    compiler = Compiler()
    lowered = compiler.lowerer.lower_manifest(manifest)
    # Validation is now component-specific; keep the call for pipeline parity.
    compiler.validator.validate_manifest(lowered, print_plan=True)

    if dry_run:
        return {"target": name}

    return ExperimentRunner(lowered).run(target_name=name, manifest_path=manifest_path)


class ExperimentRunner:
    """Unified runner for manifest v2 targets."""

    def __init__(self, manifest: Manifest) -> None:
        self.manifest = manifest

    def run(
        self,
        target_name: str,
        *,
        manifest_path: Path | None = None,
    ) -> dict[str, Path]:
        target = self._find_target(target_name)
        if isinstance(target, ProcessTargetConfig):
            from caramba.agent.process_runner import run_process_target  # local import

            result = run_process_target(
                manifest=self.manifest,
                target=target,
                manifest_path=manifest_path,
            )
            if isinstance(result, dict):
                artifacts = result.get("artifacts", None)
                if isinstance(artifacts, dict):
                    # Normalize to dict[str, Path] for caller parity.
                    return cast(dict[str, Path], artifacts)
            return {}

        assert isinstance(target, ExperimentTargetConfig)
        best_effort = bool(getattr(getattr(self.manifest.defaults, "runtime", object()), "best_effort", False))
        readiness = check_target_readiness(self.manifest, target, best_effort=best_effort)
        if readiness.errors:
            raise RuntimeError(
                "Runtime readiness check failed:\n" + format_readiness_report(readiness)
            )
        for w in readiness.warnings:
            # Best-effort mode: warn loudly when performance backends are missing.
            if best_effort and w.code in {"metal_build_tools_missing", "triton_missing"}:
                logger.fallback_warning(
                    "WARNING: Running unoptimized PyTorch fallback for DBA Decode. Performance will be degraded."
                )
            logger.warning(w.message)

        engine = TorchEngine()
        logger.header("Target", f"{target.name} ({target.type})")
        return cast(dict[str, Path], engine.run_experiment(self.manifest, target))

    def run_all(self, *, manifest_path: Path | None = None) -> dict[str, dict[str, Path]]:
        """Run every target in the manifest in declaration order."""
        results: dict[str, dict[str, Path]] = {}
        for t in self.manifest.targets:
            name = str(t.name)
            artifacts = self.run(target_name=name, manifest_path=manifest_path)
            results[name] = dict(artifacts or {})
        return results

    def _find_target(self, name: str) -> TargetConfig:
        for t in self.manifest.targets:
            if t.name == name:
                return t
        raise ValueError(f"Target '{name}' not found in manifest")
