"""Manifest targets: runnable units declared in YAML.

Targets replace the older (model + groups) coupling and allow the platform to
support arbitrary workflows without assuming "language model" as the default.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from caramba.config.agents import AgentProcessConfig, AgentTeamConfig
from caramba.config.benchmark import BenchmarkSpec
from caramba.config.component import ComponentSpec
from caramba.config.model import ModelConfig
from caramba.config.run import Run


class ExperimentTargetConfig(BaseModel):
    """A runnable experiment target.

    The target declares intent via components (task/data/system/trainer/metrics)
    and supplies concrete runs (steps/seeds/train config).
    """

    type: Literal["experiment"] = "experiment"
    name: str
    description: str = ""
    backend: str = "torch"

    task: ComponentSpec
    data: ComponentSpec
    system: ComponentSpec
    objective: ComponentSpec
    trainer: ComponentSpec

    # Optional post-run evaluators/metrics; concrete implementations live behind refs.
    metrics: list[ComponentSpec] = Field(default_factory=list)

    # Optional legacy benchmark suite (used by migrated LM presets initially).
    benchmarks: list[BenchmarkSpec] | None = None

    runs: list[Run] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_known_components(self) -> "ExperimentTargetConfig":
        # Validate built-in system types early for good errors.
        if self.system.ref in ("system.language_model", "system.generic"):
            model_payload = self.system.config.get("model", None)
            if not isinstance(model_payload, dict):
                raise ValueError(
                    f"{self.system.ref} requires system.config.model to be a dict"
                )
            # Strict parse: will raise ValidationError for unknown/extra fields.
            _ = ModelConfig.model_validate(model_payload)

        # Validate built-in dataset type shape.
        if self.data.ref == "dataset.tokens":
            p = self.data.config.get("path", None)
            bs = self.data.config.get("block_size", None)
            if not isinstance(p, str) or not p:
                raise ValueError("dataset.tokens requires data.config.path (non-empty string)")
            if not isinstance(bs, int) or bs <= 0:
                raise ValueError("dataset.tokens requires data.config.block_size (positive int)")

        return self


class ProcessTargetConfig(BaseModel):
    """A runnable agent process target."""

    type: Literal["process"] = "process"
    name: str
    description: str = ""

    team: AgentTeamConfig
    process: AgentProcessConfig


TargetConfig: TypeAlias = Annotated[
    ExperimentTargetConfig | ProcessTargetConfig,
    Field(discriminator="type"),
]

