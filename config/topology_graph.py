"""Graph topology configuration (named ports).

This is *not* the same as `config/topology.py` (which models residual/stacked
layer trees over a single tensor stream). Graph topologies operate on a
TensorDict with named keys, enabling arbitrary DAGs (U-Nets, multi-modal fusion,
Siamese nets, etc.).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GraphNodeConfig(BaseModel):
    id: str
    op: str
    # A node may read multiple inputs.
    in_keys: str | list[str] = Field(alias="in")
    # A node may write multiple outputs.
    out_keys: str | list[str] = Field(alias="out")
    config: dict[str, Any] = Field(default_factory=dict)
    repeat: int = 1


class GraphTopologyConfig(BaseModel):
    type: str = "GraphTopology"
    nodes: list[GraphNodeConfig] = Field(default_factory=list)

