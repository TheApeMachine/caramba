"""Schema loader.

Loads schema definitions from YAML into strongly-typed dataclasses.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from adapter.schema.base import (
    AttentionSchema,
    BlockSchema,
    EmbedderSchema,
    HeadSchema,
    MlpSchema,
    StateDictSchema,
)


class SchemaLoader:
    """Load schema objects from YAML files."""
    def load(self, *, path: str) -> StateDictSchema:
        """Load a schema from a YAML file path."""
        p = Path(str(path))

        if not p.exists():
            raise FileNotFoundError(f"Schema file not found: {p}")

        payload = yaml.safe_load(p.read_text(encoding="utf-8"))

        if not isinstance(payload, dict):
            raise TypeError(f"Schema YAML must be a mapping, got {type(payload).__name__} from {p}")

        return self.mapping(mapping=payload, name=str(payload.get("name", p.stem)))

    def mapping(self, *, mapping: dict[str, Any], name: str) -> StateDictSchema:
        """Load a schema from a mapping."""
        prefix = str(self.require(mapping, "prefix"))
        embedder = self.embedder(mapping=self.require_mapping(mapping, "embedder"))
        block = self.block(mapping=self.require_mapping(mapping, "block"))
        attention = self.attention(mapping=self.require_mapping(mapping, "attention"))
        mlp = self.mlp(mapping=self.require_mapping(mapping, "mlp"))
        final_norm_weight = str(self.require(mapping, "final_norm_weight"))
        head_raw = mapping.get("head", None)
        head = None

        if isinstance(head_raw, dict):
            head = HeadSchema(weight=str(self.require(head_raw, "weight")))

        return StateDictSchema(
            name=str(name),
            prefix=prefix,
            embedder=embedder,
            block=block,
            attention=attention,
            mlp=mlp,
            final_norm_weight=final_norm_weight,
            head=head,
        )

    def embedder(self, *, mapping: dict[str, Any]) -> EmbedderSchema:
        """Load embedder schema."""
        return EmbedderSchema(tokens_weight=str(self.require(mapping, "tokens_weight")))

    def block(self, *, mapping: dict[str, Any]) -> BlockSchema:
        """Load per-block schema."""
        return BlockSchema(
            path=str(self.require(mapping, "path")),
            input_norm_weight=str(self.require(mapping, "input_norm_weight")),
            post_attn_norm_weight=str(self.require(mapping, "post_attn_norm_weight")),
        )

    def attention(self, *, mapping: dict[str, Any]) -> AttentionSchema:
        """Load attention schema."""
        return AttentionSchema(
            path=str(self.require(mapping, "path")),
            q_weight=str(self.require(mapping, "q_weight")),
            k_weight=str(self.require(mapping, "k_weight")),
            v_weight=str(self.require(mapping, "v_weight")),
            o_weight=str(self.require(mapping, "o_weight")),
        )

    def mlp(self, *, mapping: dict[str, Any]) -> MlpSchema:
        """Load MLP schema."""
        return MlpSchema(
            path=str(self.require(mapping, "path")),
            gate_weight=str(self.require(mapping, "gate_weight")),
            up_weight=str(self.require(mapping, "up_weight")),
            down_weight=str(self.require(mapping, "down_weight")),
            gate_bias=self.optional_string(mapping, "gate_bias"),
            up_bias=self.optional_string(mapping, "up_bias"),
            down_bias=self.optional_string(mapping, "down_bias"),
        )

    def require(self, mapping: dict[str, Any], key: str) -> Any:
        """Require a key in a mapping."""
        if key not in mapping:
            raise KeyError(f"Schema missing required key {key!r}")
        return mapping[key]

    def require_mapping(self, mapping: dict[str, Any], key: str) -> dict[str, Any]:
        """Require a nested mapping."""
        value = self.require(mapping, key)
        if not isinstance(value, dict):
            raise TypeError(f"Schema key {key!r} must be a mapping, got {type(value).__name__}")
        return value

    def optional_string(self, mapping: dict[str, Any], key: str) -> str | None:
        """Optional string value."""
        value = mapping.get(key, None)
        if value is None:
            return None
        return str(value)

