"""Graph topology configuration (named ports).

This is *not* the same as `config/topology.py` (which models residual/stacked
layer trees over a single tensor stream). Graph topologies operate on a
TensorDict with named keys, enabling arbitrary DAGs (U-Nets, multi-modal fusion,
Siamese nets, etc.).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from config.layer import LayerConfig


class GraphNodeConfig(BaseModel):
    """A single node/layer inside a GraphTopology.

    GraphTopology is a named-port DAG over a TensorDict. In graph language these
    are often called "nodes", but structurally they function like "layers":
    they consume input keys and produce output keys.

    Two construction modes are supported:
    - `layer`: a standard `LayerConfig` (preferred for built-in layers)
    - `op` + `config`: a dynamic nn.Module factory (torch.nn.* or python:module:Symbol)

    For manifest ergonomics, layer-backed nodes may also be written inline as a
    `LayerConfig` payload (with `type: <LayerType>`) at the same level as
    `id`/`in`/`out`.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str

    # A node may read multiple inputs.
    in_keys: str | list[str] = Field(alias="in")
    # A node may write multiple outputs.
    out_keys: str | list[str] = Field(alias="out")

    repeat: int = Field(default=1, ge=1)

    # Layer-backed form.
    layer: LayerConfig | None = None

    # Op-backed form.
    op: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _lift_inline_layer_payload(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        # Inline LayerConfig form: treat remaining keys as a layer payload.
        if "layer" not in data and "op" not in data and "type" in data:
            node_keys = {"id", "in", "out", "repeat"}
            node_payload = {k: data[k] for k in node_keys if k in data}
            layer_payload = {k: v for k, v in data.items() if k not in node_keys}
            node_payload["layer"] = layer_payload
            return node_payload

        return data

    @model_validator(mode="after")
    def _validate_mode(self) -> "GraphNodeConfig":
        has_layer = self.layer is not None
        has_op = self.op is not None

        if has_layer == has_op:
            raise ValueError("GraphNodeConfig requires exactly one of 'layer' or 'op'")

        if has_layer and self.config:
            raise ValueError("GraphNodeConfig: 'config' is only valid with 'op' nodes")

        return self


class GraphTopologyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["GraphTopology"] = "GraphTopology"

    # Canonical field name is layers, to align with other topology types.
    # Graph manifests may still call these `nodes`; that's a stable alias.
    layers: list[GraphNodeConfig] = Field(
        default_factory=list,
        validation_alias=AliasChoices("layers", "nodes"),
    )

    @property
    def nodes(self) -> list[GraphNodeConfig]:
        """Backward-compatible alias for code that still expects `.nodes`."""
        return self.layers
