"""Schema definitions for state-dict adapters.

Schemas are data objects describing external checkpoint naming/layout
conventions. They are consumed by adapters and must not become schema-as-code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BlockSchema:
    """Per-block key templates."""

    path: str
    input_norm_weight: str
    post_attn_norm_weight: str


@dataclass(frozen=True, slots=True)
class AttentionSchema:
    """Attention key templates."""

    path: str
    q_weight: str
    k_weight: str
    v_weight: str
    o_weight: str


@dataclass(frozen=True, slots=True)
class MlpSchema:
    """MLP key templates."""

    path: str
    gate_weight: str
    up_weight: str
    down_weight: str
    gate_bias: str | None = None
    up_bias: str | None = None
    down_bias: str | None = None


@dataclass(frozen=True, slots=True)
class EmbedderSchema:
    """Embedder key templates."""

    tokens_weight: str


@dataclass(frozen=True, slots=True)
class HeadSchema:
    """LM head key template."""

    weight: str


@dataclass(frozen=True, slots=True)
class StateDictSchema:
    """State-dict schema.

    `prefix` is the root prefix for all keys (e.g. "model"). Block paths must
    include "{i}" for the layer index.
    """

    name: str
    prefix: str
    embedder: EmbedderSchema
    block: BlockSchema
    attention: AttentionSchema
    mlp: MlpSchema
    final_norm_weight: str
    head: HeadSchema | None = None

