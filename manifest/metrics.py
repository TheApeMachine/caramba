"""Metrics manifest

Metrics configuration is expressed as a set of backend toggles. Each backend has
an `enabled` flag and (optionally) additional fields required for that backend.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, StrictBool, model_validator


class MetricsHDF5(BaseModel):
    """HDF5 metrics configuration."""
    enabled: StrictBool = Field(
        False,
        description="Whether to enable HDF5 metrics"
    )


class MetricsTensorboard(BaseModel):
    """TensorBoard metrics configuration."""
    enabled: StrictBool = Field(
        False,
        description="Whether to enable TensorBoard metrics"
    )


class MetricsWandb(BaseModel):
    """Wandb metrics configuration."""
    enabled: StrictBool = Field(False, description="Whether to enable Wandb metrics")
    project: str = Field("", description="Wandb project")
    entity: str = Field("", description="Wandb entity")
    mode: str = Field("online", description="Wandb mode")
    eval_iters: int = Field(0, description="Wandb eval iterations")

    @model_validator(mode="after")
    def validate_required_fields(self) -> "MetricsWandb":
        """Ensure required fields are present when wandb is enabled."""
        if self.enabled and not str(self.project).strip():
            raise ValueError("wandb.project must be set when wandb.enabled is true")
        if self.eval_iters < 0:
            raise ValueError("wandb.eval_iters must be non-negative")
        return self


class MetricsLiveplot(BaseModel):
    """Liveplot metrics configuration."""
    enabled: StrictBool = Field(
        False,
        description="Whether to enable Liveplot metrics"
    )


class MetricsJSONL(BaseModel):
    """JSONL metrics configuration."""
    enabled: StrictBool = Field(
        False,
        description="Whether to enable JSONL metrics"
    )


class Metrics(BaseModel):
    """A metrics configuration.

    This mirrors the `instrumentation.metrics` section in template manifests.
    """
    hdf5: MetricsHDF5 = Field(
        default_factory=lambda: MetricsHDF5(enabled=False),
        description="HDF5 metrics"
    )

    tensorboard: MetricsTensorboard = Field(
        default_factory=lambda: MetricsTensorboard(enabled=False),
        description="TensorBoard metrics"
    )

    wandb: MetricsWandb = Field(default_factory=lambda: MetricsWandb(
        enabled=False,
        project="",
        entity="",
        mode="online",
        eval_iters=0
    ), description="Wandb metrics")

    liveplot: MetricsLiveplot = Field(
        default_factory=lambda: MetricsLiveplot(enabled=False),
        description="Liveplot metrics"
    )

    jsonl: MetricsJSONL = Field(
        default_factory=lambda: MetricsJSONL(enabled=False),
        description="JSONL metrics"
    )
