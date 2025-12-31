"""Layer configuration with discriminated unions.

Each layer type (attention, MLP, normalization) has its own config class.
Pydantic's discriminated unions allow YAML like `type: AttentionLayer` to
automatically deserialize into the correct config class.
"""
from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from caramba.config import Config, PositiveFloat, PositiveInt, Probability


class AttentionMode(str, enum.Enum):
    """Which attention variant to use.

    STANDARD: Full multi-head attention
    GQA: Grouped-query attention (shared KV heads)
    DECOUPLED: DBA with separate semantic/geometric paths
    """

    STANDARD = "standard"
    GQA = "gqa"
    DECOUPLED = "decoupled"


class LayerType(str, enum.Enum):
    """Enumeration of layer types for type-safe config parsing.

    Using an enum prevents magic strings and gives better error messages
    when an unknown layer type is specified in YAML.
    """

    LAYER_NORM = "LayerNormLayer"
    RMS_NORM = "RMSNormLayer"
    LINEAR = "LinearLayer"
    LORA_LINEAR = "LoRALinearLayer"
    DROPOUT = "DropoutLayer"
    ATTENTION = "AttentionLayer"
    SWIGLU = "SwiGLULayer"
    GLU = "GLULayer"
    MOE = "MoELayer"
    SSM = "SSMLayer"
    CONV2D = "Conv2dLayer"
    RNN = "RNNLayer"
    GRAPH_CONV = "GraphConvLayer"
    DENSE = "DenseLayer"

    @classmethod
    def from_str(cls, s: str) -> "LayerType":
        """Convert a string to a LayerType."""
        return cls(s)

    @staticmethod
    def module_name() -> str:
        """Return the Python module containing layer implementations."""
        return "caramba.layer"


class LinearLayerConfig(Config):
    """Configuration for a simple linear projection."""

    type: Literal[LayerType.LINEAR] = LayerType.LINEAR
    d_in: PositiveInt
    d_out: PositiveInt
    bias: bool = True


class LoRALinearLayerConfig(Config):
    """Configuration for a LoRA-enabled linear projection."""

    type: Literal[LayerType.LORA_LINEAR] = LayerType.LORA_LINEAR
    d_in: PositiveInt
    d_out: PositiveInt
    r: PositiveInt = 8
    alpha: PositiveFloat = 16.0
    dropout: Probability = 0.0
    bias: bool = True


class LayerNormLayerConfig(Config):
    """Configuration for standard LayerNorm."""

    type: Literal[LayerType.LAYER_NORM] = LayerType.LAYER_NORM
    d_model: PositiveInt
    eps: PositiveFloat = 1e-5


class RMSNormLayerConfig(Config):
    """Configuration for RMSNorm (used in Llama and modern LLMs)."""

    type: Literal[LayerType.RMS_NORM] = LayerType.RMS_NORM
    d_model: PositiveInt
    eps: PositiveFloat = 1e-5
    elementwise_affine: bool = True


class DropoutLayerConfig(Config):
    """Configuration for dropout regularization."""

    type: Literal[LayerType.DROPOUT] = LayerType.DROPOUT
    p: Probability = 0.0


class AttentionLayerConfig(Config):
    """Configuration for unified attention (standard/GQA/DBA).

    The mode field selects the attention variant:
    - standard: Each head has its own Q/K/V projections
    - gqa: Multiple Q heads share K/V heads
    - decoupled: Separate semantic (content) and geometric (position) paths

    For DBA (decoupled), set sem_dim and geo_dim. By default RoPE is only applied
    to the geometric path; the semantic path is position-invariant. Ablations
    can enable RoPE on the semantic path as well.
    """

    type: Literal[LayerType.ATTENTION] = LayerType.ATTENTION

    # Core dimensions
    d_model: PositiveInt
    n_heads: PositiveInt
    n_kv_heads: PositiveInt | None = None

    # Attention mode
    mode: AttentionMode = AttentionMode.STANDARD

    # Optional attention bottleneck dimension
    attn_dim: PositiveInt | None = None

    # DBA dimensions (only used when mode=decoupled)
    sem_dim: PositiveInt | None = None
    geo_dim: PositiveInt | None = None

    # RoPE settings
    rope_enabled: bool = True
    rope_base: float = 10000.0
    # Optional RoPE scaling config (for Llama 3 style RoPE).
    # This is a direct, manifest-driven escape hatch: it should match the HF config's
    # `rope_scaling` dict when present.
    rope_scaling: dict[str, object] | None = None

    # DBA ablation toggles (only meaningful when mode=decoupled)
    # - rope_semantic: apply RoPE to the semantic Q/K projections too
    # - tie_qk: tie semantic W_Q and W_K (W_Q,sem == W_K,sem)
    # - null_attn: add a learned "null" KV token (sink/skip token) always available
    rope_semantic: bool = False
    tie_qk: bool = False
    null_attn: bool = False

    # DBA gating (learned per-head semantic/geometric mixing)
    decoupled_gate: bool = False
    decoupled_gate_dynamic: bool = False

    # Standard attention settings
    is_causal: bool = True
    dropout_p: Probability = 0.0
    bias: bool = False
    learned_temp: bool = False

    # Long-sequence performance knobs (optional).
    # q_chunk: compute attention in chunks over the query length to reduce peak memory.
    # local_window: restrict attention to a fixed window around each query position.
    q_chunk: PositiveInt | None = None
    local_window: PositiveInt | None = None

    # Memory summarization: optional compression of the far past.
    # When mem_block is set, we summarize the prefix (everything before the local_window)
    # into one token per block of size mem_block. The method controls how the block is summarized.
    mem_block: PositiveInt | None = None
    mem_summarize: Literal["mean", "linear", "conv"] = "mean"
    mem_activation_threshold: PositiveInt | None = None

    @property
    def head_dim(self) -> int:
        """Compute head dimension from total attention dimension."""
        dim = self.attn_dim if self.attn_dim is not None else self.d_model
        return dim // self.n_heads

    @property
    def kv_heads(self) -> int:
        """Number of KV heads (for GQA)."""
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

    @property
    def sem_head_dim(self) -> int | None:
        """Per-head semantic dimension for DBA."""
        if self.sem_dim is None:
            return None
        return self.sem_dim // self.n_heads

    @property
    def geo_head_dim(self) -> int | None:
        """Per-head geometric dimension for DBA."""
        if self.geo_dim is None:
            return None
        return self.geo_dim // self.n_heads

    @property
    def v_dim(self) -> int:
        """Value projection dimension."""
        return self.attn_dim if self.attn_dim is not None else self.d_model


class SwiGLULayerConfig(Config):
    """Configuration for SwiGLU MLP (gate/up/down projections)."""

    type: Literal[LayerType.SWIGLU] = LayerType.SWIGLU
    d_model: PositiveInt
    d_ff: PositiveInt
    bias: bool = True


class GLULayerConfig(Config):
    """Configuration for generic Gated Linear Unit (GLU)."""

    type: Literal[LayerType.GLU] = LayerType.GLU
    d_model: PositiveInt
    d_ff: PositiveInt
    activation: str = "silu"
    bias: bool = True


class MoELayerConfig(Config):
    """Configuration for Mixture of Experts (MoE) layer."""

    type: Literal[LayerType.MOE] = LayerType.MOE
    d_model: PositiveInt
    num_experts: PositiveInt = 8
    top_k: PositiveInt = 2
    d_ff: PositiveInt
    bias: bool = True
    gate_requires_grad: bool = True
    up_requires_grad: bool = True
    down_requires_grad: bool = True
    load_balancing: bool = False
    load_balancing_loss_weight: PositiveFloat = 1.0
    load_balancing_loss_type: Literal["kl", "mse"] = "kl"
    load_balancing_loss_temperature: PositiveFloat = 1.0
    load_balancing_loss_temperature_schedule: Literal["linear", "cosine"] = "linear"
    load_balancing_loss_temperature_schedule_params: dict[str, float] = Field(default_factory=dict)


class SSMLayerConfig(Config):
    """Configuration for State Space Model (SSM) layer."""

    type: Literal[LayerType.SSM] = LayerType.SSM
    d_model: PositiveInt
    d_state: PositiveInt = 16
    d_conv: PositiveInt = 4
    expand: PositiveInt = 2
    dt_rank: str | int = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    bias: bool = False
    conv_bias: bool = True


class Conv2dLayerConfig(Config):
    """Configuration for 2D Convolution layer."""

    type: Literal[LayerType.CONV2D] = LayerType.CONV2D
    in_channels: PositiveInt
    out_channels: PositiveInt
    kernel_size: PositiveInt | tuple[PositiveInt, PositiveInt]
    stride: PositiveInt | tuple[PositiveInt, PositiveInt] = 1
    padding: int | tuple[int, int] | str = 0
    dilation: PositiveInt | tuple[PositiveInt, PositiveInt] = 1
    groups: PositiveInt = 1
    bias: bool = True
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros"


class RNNLayerConfig(Config):
    """Configuration for Recurrent Neural Network layer.

    Supports standard LSTM/GRU and optimized variants.
    """

    type: Literal[LayerType.RNN] = LayerType.RNN
    cell_type: Literal["lstm", "gru", "rnn_tanh", "rnn_relu"] = "lstm"
    input_size: PositiveInt
    hidden_size: PositiveInt
    num_layers: PositiveInt = 1
    bias: bool = True
    batch_first: bool = True
    dropout: Probability = 0.0
    bidirectional: bool = False
    proj_size: PositiveInt = 0  # LSTM only


class GraphConvLayerConfig(Config):
    """Configuration for Graph Convolution layer.

    Supports GCN and GAT variants.
    """

    type: Literal[LayerType.GRAPH_CONV] = LayerType.GRAPH_CONV
    kind: Literal["gcn", "gat"] = "gcn"
    in_features: PositiveInt
    out_features: PositiveInt
    bias: bool = True
    # GAT specific
    heads: PositiveInt = 1
    concat: bool = True
    negative_slope: PositiveFloat = 0.2
    dropout: Probability = 0.0


class DenseLayerConfig(Config):
    """Configuration for Dense (MLP) layer.

    Includes optional normalization, activation, and dropout.
    """

    type: Literal[LayerType.DENSE] = LayerType.DENSE
    d_in: PositiveInt
    d_out: PositiveInt
    bias: bool = True
    activation: str | None = None  # e.g. "relu", "gelu", "silu"
    normalization: Literal["layer_norm", "rms_norm"] | None = None
    dropout: Probability = 0.0


# Union type for any layer config, with automatic deserialization
LayerConfig: TypeAlias = Annotated[
    LinearLayerConfig
    | LoRALinearLayerConfig
    | LayerNormLayerConfig
    | RMSNormLayerConfig
    | DropoutLayerConfig
    | AttentionLayerConfig
    | SwiGLULayerConfig
    | GLULayerConfig
    | MoELayerConfig
    | SSMLayerConfig
    | Conv2dLayerConfig
    | RNNLayerConfig
    | GraphConvLayerConfig
    | DenseLayerConfig,
    Field(discriminator="type"),
]
