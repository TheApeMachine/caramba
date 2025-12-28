"""Generic component references for manifest-driven composition.

The platform needs to remain resilient to future developments. Instead of
hard-coding a fixed set of first-class config unions for every possible
technique, the manifest can reference components by a stable semantic id
(`ref`) and optionally specify an implementation override (`impl`).

Resolution is handled by the runtime registry/engine.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class ComponentSpec(BaseModel):
    """Reference to a runnable/buildable component.

    - **ref**: stable semantic identifier (e.g. `task.language_modeling`)
    - **impl**: optional implementation selector. Supports:
      - backend keys (e.g. `torch`, `jax`, `sklearn`) handled by the engine
      - escape hatch: `python:some.module:SymbolName`
    - **config**: free-form config payload forwarded to the implementation.
    """

    ref: str
    impl: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_str(cls, v: object) -> object:
        if isinstance(v, str):
            return {"ref": v}
        return v

