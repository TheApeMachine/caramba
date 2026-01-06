"""Manifest: the top-level configuration file.

Manifest v2 is *target-based* and intentionally model-agnostic:
- A manifest is a collection of runnable targets (experiments or agent processes).
- Targets reference components by semantic ids (task/data/system/trainer/metrics).

This keeps intent (manifest) separate from implementation (registry/engine),
aligning with `internal/CORE_PHILOSOPHY.md`.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from caramba.config import PositiveInt
from caramba.config.defaults import Defaults
from caramba.config.resolve import Resolver
from caramba.config.target import TargetConfig


class Manifest(BaseModel):
    """The complete manifest specification loaded from YAML/JSON."""

    version: PositiveInt
    name: str | None = None
    notes: str = ""
    # Optional override for where artifacts are written.
    # Default is "artifacts" to preserve legacy behavior.
    artifacts_dir: str = "artifacts"
    defaults: Defaults

    # Runnable units.
    targets: list[TargetConfig] = Field(default_factory=list)

    # Optional named targets. If present, entrypoints["default"] is used when
    # the CLI doesn't specify --target. Values can be either a bare target name
    # or an explicit `target:<name>` string.
    entrypoints: dict[str, str] | None = None

    @classmethod
    def from_path(cls, path: Path) -> "Manifest":
        """Load and validate a manifest from a JSON or YAML file.

        Supports variable substitution via a `vars` section at the top level.
        Variables can be referenced as `${var_name}` throughout the config.
        """
        text = path.read_text(encoding="utf-8")
        match path.suffix.lower():
            case ".json":
                payload = json.loads(text)
            case ".yml" | ".yaml":
                payload = yaml.safe_load(text)
            case s:
                raise ValueError(f"Unsupported format '{s}'")

        if payload is None:
            raise ValueError("Manifest payload is empty.")
        if not isinstance(payload, dict):
            raise ValueError(f"Manifest payload must be a dict, got {type(payload)!r}")

        # Process variable substitution
        vars_payload = payload.pop("vars", None)
        if vars_payload is not None:
            if not isinstance(vars_payload, dict):
                raise ValueError(
                    f"Manifest vars must be a dict, got {type(vars_payload)!r}"
                )
            payload = Resolver(vars_payload).resolve(payload)

        return cls.model_validate(payload)
