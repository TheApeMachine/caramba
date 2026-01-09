"""Manifest builder for constructing experiment configurations.

Provides utilities for AI agents to construct valid manifest YAML files
based on research goals and experimental parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from caramba.config.manifest import Manifest
from caramba.console import logger


class ExperimentSpec(BaseModel):
    """Specification for an experiment to include in a manifest."""

    name: str = Field(description="Name of the experiment target")
    description: str = Field(description="What this experiment tests")
    model_type: str = Field(description="Type of model (e.g., transformer, mamba)")
    dataset: str = Field(description="Dataset to use")
    training_config: dict[str, Any] = Field(default_factory=dict, description="Training parameters")
    metrics: list[str] = Field(default_factory=list, description="Metrics to collect")


class ManifestSpec(BaseModel):
    """Specification for a complete manifest."""

    name: str = Field(description="Name of the manifest")
    notes: str = Field(description="Description of the experiment suite")
    experiments: list[ExperimentSpec] = Field(description="List of experiments to run")


class ManifestBuilder:
    """Builder for constructing manifest YAML files.

    Provides methods to create valid manifest configurations that can be
    run through the experiment runner.
    """

    def __init__(self, presets_dir: str = "config/presets") -> None:
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    def build_from_spec(self, spec: ManifestSpec) -> dict[str, Any]:
        """Build a manifest dict from a specification."""
        manifest = {
            "version": 2,
            "name": spec.name,
            "notes": spec.notes,
            "defaults": {
                "data": {"tokenizer": "llama"},
                "logging": {"instrument": "rich"},
                "runtime": {"save_every": 500},
            },
            "targets": [],
        }

        for exp in spec.experiments:
            target = self._build_experiment_target(exp)
            manifest["targets"].append(target)

        return manifest

    def _build_experiment_target(self, exp: ExperimentSpec) -> dict[str, Any]:
        """Build a single experiment target configuration."""
        target = {
            "type": "experiment",
            "name": exp.name,
            "description": exp.description,
            "task": {"type": "pretrain"},
            "data": {
                "type": exp.dataset,
                "batch_size": exp.training_config.get("batch_size", 32),
                "seq_len": exp.training_config.get("seq_len", 512),
            },
            "system": {
                "type": exp.model_type,
            },
            "trainer": {
                "type": "standard",
                "max_steps": exp.training_config.get("max_steps", 1000),
                "learning_rate": exp.training_config.get("learning_rate", 1e-4),
            },
            "metrics": {
                "types": exp.metrics or ["loss", "perplexity"],
            },
        }
        return target

    def save_manifest(self, manifest: dict[str, Any], filename: str) -> Path:
        """Save a manifest to the presets directory."""
        if not filename.endswith((".yml", ".yaml")):
            filename = f"{filename}.yml"

        path = self.presets_dir / filename
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

        logger.success(f"Saved manifest to {path}")
        return path

    def validate_manifest(self, manifest: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate a manifest configuration.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            Manifest.model_validate(manifest)
            return True, None
        except Exception as e:
            return False, str(e)

    def list_existing_presets(self) -> list[str]:
        """List all existing manifest presets."""
        return [p.name for p in self.presets_dir.glob("*.yml")]

    def load_preset(self, name: str) -> dict[str, Any]:
        """Load an existing preset manifest."""
        path = self.presets_dir / name
        if not path.suffix:
            path = path.with_suffix(".yml")
        if not path.exists():
            raise FileNotFoundError(f"Preset not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def clone_and_modify(
        self,
        base_preset: str,
        new_name: str,
        modifications: dict[str, Any],
    ) -> dict[str, Any]:
        """Clone an existing preset and apply modifications."""
        base = self.load_preset(base_preset)
        manifest = self._deep_merge(base, modifications)
        manifest["name"] = new_name
        return manifest

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
