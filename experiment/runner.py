"""Experiment orchestration (manifest v2).

Manifest v2 is target-based:
- targets are runnable units (experiments or agent processes)
- experiments are executed via an Engine (backend) which resolves components
  through the registry.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from compiler import Compiler
from config.agents import AgentProcessConfig
from config.manifest import Manifest
from config.target import ExperimentTargetConfig, ProcessTargetConfig, TargetConfig
from console import logger
from runtime.engine.torch_engine import TorchEngine
from runtime.engine.lightning_engine import LightningEngine
from runtime.engine import get_mlx_engine
from orchestrator.compute.vast_ai import VastAIClient
from runtime.readiness import check_target_readiness, format_readiness_report
from ai.agent import Agent
from ai.persona import PersonaLoader


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
            process_type = target.process.type

            # Build agents from the declared team mapping (role_key -> persona yaml name).
            loader = PersonaLoader()
            agents: dict[str, Agent] = {}
            for role_key, persona_name in target.team.root.items():
                try:
                    persona = loader.load(persona_name)
                except FileNotFoundError as e:
                    raise FileNotFoundError(
                        f"Persona file not found for role '{role_key}': {e}"
                    ) from e
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load persona '{persona_name}': {e}"
                    ) from e
                agents[role_key] = Agent(persona=persona)

            return {}

        assert isinstance(target, ExperimentTargetConfig)
        readiness = check_target_readiness(self.manifest, target)
        if readiness.errors:
            raise RuntimeError(
                "Runtime readiness check failed:\n" + format_readiness_report(readiness)
            )
        for w in readiness.warnings:
            logger.warning(w.message)

        # Handle remote compute provisioning
        if hasattr(target, "compute") and target.compute.type == "vast_ai":
            logger.header("Remote Provisioning", f"Vast.ai ({target.compute.gpu_name})")
            client = VastAIClient(api_key=getattr(self.manifest.defaults.compute, "vast_ai_api_key", None))
            ssh_str = client.run_lifecycle(target.compute)
            if not ssh_str:
                raise RuntimeError("Failed to provision Vast.ai instance.")

            # In a full implementation, we would now:
            # 1. Sync the codebase to the remote instance (via ssh_str)
            # 2. Trigger the run remotely via SSH
            # 3. Stream back logs
            # 4. client.decommission_instance(...)
            logger.info(f"Instance ready. Remote run not fully implemented, continuing locally for demonstration.")

        # Engine selection
        backend = str(target.backend)
        if backend == "lightning":
            engine = LightningEngine()
        elif backend in {"torch", "pt"}:
            engine = TorchEngine()
        elif backend == "mlx":
            engine = get_mlx_engine()
        else:
            raise ValueError(
                f"Unsupported target.backend={backend!r}; expected one of 'lightning', 'torch', 'pt', 'mlx'"
            )

        logger.header("Target", f"{target.name} ({target.type}) • backend={target.backend}")
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
