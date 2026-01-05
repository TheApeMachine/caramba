"""Llama checkpoint loading with DBA attention surgery.

When upcycling a Llama model to DBA, we can't just copy weights—the attention
architecture is different. Standard attention has Q/K/V projections; DBA has
separate semantic and geometric Q/K paths. This module handles the "surgery"
of initializing DBA projections from the original Llama weights.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from caramba.config.layer import AttentionMode
from caramba.carmath import randomized_svd
from caramba.layer.attention import AttentionLayer
from caramba.layer.linear import LinearLayer
from caramba.layer.rms_norm import RMSNormLayer
from caramba.layer.swiglu import SwiGLULayer
from caramba.loader.state_reader import StateReader
from caramba.model.embedder import Embedder
from caramba.console import logger


class LlamaUpcycle:
    """Loads Llama checkpoints into caramba models, handling architecture differences.

    For standard/GQA attention, weights copy directly. For DBA attention,
    we perform "attention surgery" using SVD to split the teacher's Q/K
    into semantic (content routing) and geometric (positional) components.
    """

    def __init__(
        self,
        model: nn.Module,
        state_dict: dict[str, Tensor],
        prefix: str = "model",
        head_key: str = "lm_head.weight",
        *,
        dba_init: str = "svd",
    ) -> None:
        """Set up the loader with a target model and source weights.

        Args:
            model: The caramba model to load weights into
            state_dict: Llama checkpoint weights
            prefix: Key prefix in the state_dict (usually "model")
            head_key: Key for the LM head weight
        """
        self.model = model
        self.state = StateReader(state_dict)
        self.prefix = prefix
        self.head_key = head_key
        self.dba_init = str(dba_init).lower().strip()
        if self.dba_init not in {"svd", "random"}:
            raise ValueError(f"Unsupported dba_init={dba_init!r} (expected 'svd' or 'random')")

    def apply(self) -> None:
        """Load all weights from the Llama checkpoint into the model.

        Handles embeddings, all transformer blocks (attention, MLP, norms),
        the final norm, and the LM head.
        """
        attn = self.collect(AttentionLayer)
        mlp = self.collect(SwiGLULayer)
        norms = self.collect(RMSNormLayer)

        if len(attn) != len(mlp):
            raise ValueError(
                f"Attention/MLP count mismatch: {len(attn)} vs {len(mlp)}"
            )
        if len(norms) != 2 * len(attn) + 1:
            raise ValueError(
                f"Expected {2 * len(attn) + 1} norms, got {len(norms)}"
            )

        self.load_embedder()
        self.load_blocks(attn=attn, mlp=mlp, norms=norms)
        self.load_final_norm(norms[-1])
        self.load_head()

    def collect(self, kind: type[nn.Module]) -> list[nn.Module]:
        """Find all modules of a given type in traversal order."""
        return [m for _, m in self.model.named_modules() if isinstance(m, kind)]

    def load_embedder(self) -> None:
        """Load the token embedding table."""
        embed = self.find_embedder()
        if embed.token_embedding is None:
            raise ValueError("Model has no embedder with token_embedding")

        key = self.state.key(self.prefix, "embed_tokens", "weight")
        weight = self.state.get(key)
        embed.token_embedding.weight.data.copy_(weight)

    def load_blocks(
        self,
        attn: list[nn.Module],
        mlp: list[nn.Module],
        norms: list[nn.Module],
    ) -> None:
        """Load all transformer blocks: attention, MLP, and layer norms."""
        for idx, (att, mlp_layer) in enumerate(zip(attn, mlp)):
            layer_prefix = self.state.key(self.prefix, "layers", str(idx))

            self.load_rms_norm(norms[2 * idx], layer_prefix, "input_layernorm")
            self.load_rms_norm(
                norms[2 * idx + 1], layer_prefix, "post_attention_layernorm"
            )
            self.load_attention(att, layer_prefix)
            self.load_mlp(mlp_layer, layer_prefix)

    def load_final_norm(self, norm: nn.Module) -> None:
        """Load the final RMSNorm before the LM head."""
        key = self.state.key(self.prefix, "norm", "weight")
        if isinstance(norm, RMSNormLayer) and norm.weight is not None:
            norm.weight.data.copy_(self.state.get(key))

    def load_head(self) -> None:
        """Load the LM head (output projection to vocabulary).

        Falls back to using the embedding weights if lm_head.weight is missing
        (tied embeddings).
        """
        head = self.find_head()

        weight = self.state.get_optional(self.head_key)
        if weight is None:
            embed_key = self.state.key(self.prefix, "embed_tokens", "weight")
            weight = self.state.get(embed_key)

        head.linear.weight.data.copy_(weight)

    def load_rms_norm(self, layer: nn.Module, layer_prefix: str, name: str) -> None:
        """Load RMSNorm scale weights."""
        if not isinstance(layer, RMSNormLayer) or layer.weight is None:
            raise TypeError(
                "RMSNorm load failed: unexpected module type or missing weight.\n"
                f"Expected RMSNormLayer with weight, got {type(layer).__name__}.\n"
                f"Context: layer_prefix={layer_prefix!r} name={name!r}"
            )
        key = self.state.key(layer_prefix, name, "weight")
        layer.weight.data.copy_(self.state.get(key))

    def load_attention(self, layer: nn.Module, layer_prefix: str) -> None:
        """Load attention weights, using DBA surgery for decoupled mode.

        For standard attention, weights copy directly. For DBA, we use SVD
        to initialize the semantic and geometric projections.
        """
        if not isinstance(layer, AttentionLayer):
            raise TypeError(
                "Attention load failed: unexpected module type.\n"
                f"Expected AttentionLayer, got {type(layer).__name__}.\n"
                f"Context: layer_prefix={layer_prefix!r}"
            )

        attn_prefix = self.state.key(layer_prefix, "self_attn")

        q_weight = self.state.get(self.state.key(attn_prefix, "q_proj", "weight"))
        k_weight = self.state.get(self.state.key(attn_prefix, "k_proj", "weight"))
        v_weight = self.state.get(self.state.key(attn_prefix, "v_proj", "weight"))
        o_weight = self.state.get(self.state.key(attn_prefix, "o_proj", "weight"))

        if layer.mode == AttentionMode.DECOUPLED:
            logger.info("Loading DBA attention")
            self._load_attention_dba(layer, attn_prefix, q_weight, k_weight, v_weight, o_weight)
        else:
            logger.info("Loading standard attention")
            self._load_attention_standard(layer, q_weight, k_weight, v_weight, o_weight)

    def _load_attention_standard(
        self,
        layer: AttentionLayer,
        q_weight: Tensor,
        k_weight: Tensor,
        v_weight: Tensor,
        o_weight: Tensor,
    ) -> None:
        """Copy weights directly for standard/GQA attention."""
        if layer.q_proj is None or layer.k_proj is None:
            raise ValueError("Standard attention layer missing Q/K projections")

        layer.q_proj.weight.data.copy_(q_weight)
        layer.k_proj.weight.data.copy_(k_weight)
        layer.v_proj.weight.data.copy_(v_weight)
        layer.out_proj.weight.data.copy_(o_weight)

    def _load_attention_dba(
        self,
        layer: AttentionLayer,
        attn_prefix: str,
        q_weight: Tensor,
        k_weight: Tensor,
        v_weight: Tensor,
        o_weight: Tensor,
    ) -> None:
        """Initialize DBA projections from teacher attention using SVD.

        The key insight is that semantic attention (content routing) is
        low-rank, while geometric attention (positional structure) captures
        remaining variance. We use SVD to separate these components.

        - Semantic Q/K: Top singular vectors (content/topic routing)
        - Geometric Q/K: Remaining singular vectors (position patterns)
        - Gate: Initialize to 0.5 (balanced weighting)
        - V and O: Copy directly (same dimension)
        """
        if layer.q_sem is None or layer.k_sem is None:
            raise ValueError("DBA layer missing semantic projections")
        if layer.q_geo is None or layer.k_geo is None:
            raise ValueError("DBA layer missing geometric projections")

        sem_dim = layer.q_sem.out_features
        geo_dim = layer.q_geo.out_features

        # V and O: copy directly (may truncate/pad for different v_dim)
        v_out_dim = layer.v_proj.out_features
        if v_weight.size(0) == v_out_dim:
            layer.v_proj.weight.data.copy_(v_weight)
        else:
            copy_dim = min(v_weight.size(0), v_out_dim)
            layer.v_proj.weight.data[:copy_dim, :].copy_(v_weight[:copy_dim, :])

        o_in_dim = layer.out_proj.in_features
        if o_weight.size(1) == o_in_dim:
            layer.out_proj.weight.data.copy_(o_weight)
        else:
            copy_dim = min(o_weight.size(1), o_in_dim)
            layer.out_proj.weight.data[:, :copy_dim].copy_(o_weight[:, :copy_dim])

        # SVD decomposition for Q
        if self.dba_init == "svd":
            self._init_dba_projection_from_svd(
                layer.q_sem.weight,
                layer.q_geo.weight,
                q_weight,
                sem_dim,
                geo_dim,
                seed=f"{attn_prefix}.q",
            )

        # SVD decomposition for K
        if self.dba_init == "svd":
            self._init_dba_projection_from_svd(
                layer.k_sem.weight,
                layer.k_geo.weight,
                k_weight,
                sem_dim,
                geo_dim,
                seed=f"{attn_prefix}.k",
            )

        # Initialize gate to balanced (sigmoid(0) = 0.5)
        if layer.decoupled_gate_logit is not None:
            layer.decoupled_gate_logit.data.zero_()

    def _init_dba_projection_from_svd(
        self,
        sem_weight: Tensor,
        geo_weight: Tensor,
        teacher_weight: Tensor,
        sem_dim: int,
        geo_dim: int,
        *,
        seed: str | None = None,
    ) -> None:
        """Split teacher projection into semantic and geometric using SVD.

        The teacher's Q or K matrix is decomposed as U @ S @ Vh. The top
        singular vectors (semantic) capture content routing patterns; the
        remaining vectors (geometric) capture positional structure.
        """
        # We only need the leading sem_dim+geo_dim singular components (DBA bottleneck),
        # and we only need the first sem_dim/geo_dim rows of the reconstructions.
        # Use a randomized truncated SVD on the model device for speed.
        dev = sem_weight.device
        A = teacher_weight.to(device=dev, dtype=torch.float32)

        target_rank = int(sem_dim) + int(geo_dim)
        if target_rank <= 0:
            raise ValueError(f"Invalid DBA target_rank={target_rank} (sem_dim={sem_dim}, geo_dim={geo_dim})")

        # If the target rank is close to full rank, fall back to exact SVD.
        full_rank = min(int(A.shape[0]), int(A.shape[1]))
        use_exact = target_rank >= full_rank

        try:
            if use_exact:
                U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            else:
                U, S, Vh = randomized_svd(A, rank=target_rank, n_iter=2, oversample=8, seed=seed)
        except Exception as e:
            raise RuntimeError(
                "DBA SVD initialization failed.\n"
                "Why this matters: dba_init=svd is a strict part of the attention-surgery contract; "
                "we do not silently fall back to a different initialization.\n"
                "Fix:\n"
                "  - Reduce sem_dim/geo_dim (target_rank = sem_dim + geo_dim)\n"
                "  - Use a smaller model or run on a device with more memory\n"
                "  - Or set train.dba_init=random to explicitly opt into random init\n"
                f"Context: seed={seed!r} teacher_weight_shape={tuple(teacher_weight.shape)} "
                f"sem_dim={int(sem_dim)} geo_dim={int(geo_dim)} device={dev.type}\n"
                f"Error: {type(e).__name__}: {e}"
            ) from e

        rank = min(int(S.size(0)), target_rank)
        sem_rank = min(int(sem_dim), rank)
        geo_rank = min(int(geo_dim), max(0, rank - int(sem_dim)))

        # Semantic: first sem_rank components, but only first sem_dim rows.
        if sem_rank > 0:
            u_rows = U[: int(sem_dim), :sem_rank]
            sem = (u_rows * S[:sem_rank].view(1, -1)) @ Vh[:sem_rank, :]
            sem_weight.data.copy_(sem.to(dtype=sem_weight.dtype))

        # Geometric: next geo_rank components, only first geo_dim rows.
        if geo_rank > 0:
            geo_start = sem_rank
            geo_end = sem_rank + geo_rank
            u_rows = U[: int(geo_dim), geo_start:geo_end]
            geo = (u_rows * S[geo_start:geo_end].view(1, -1)) @ Vh[geo_start:geo_end, :]
            geo_weight.data.copy_(geo.to(dtype=geo_weight.dtype))

    def load_mlp(self, layer: nn.Module, layer_prefix: str) -> None:
        """Load SwiGLU MLP weights (gate, up, down projections).

        Fuses the teacher's gate and up projections into the caramba model's
        fused w_gate_up layer.
        """
        if not isinstance(layer, SwiGLULayer):
            raise TypeError(
                "SwiGLU load failed: unexpected module type.\n"
                f"Expected SwiGLULayer, got {type(layer).__name__}.\n"
                f"Context: layer_prefix={layer_prefix!r}"
            )

        mlp_prefix = self.state.key(layer_prefix, "mlp")

        # Load gate and up projections from teacher
        gate_weight = self.state.get(self.state.key(mlp_prefix, "gate_proj", "weight"))
        up_weight = self.state.get(self.state.key(mlp_prefix, "up_proj", "weight"))

        # Fuse them into w_gate_up
        # gate_up weight is (2 * d_ff, d_model)
        # We concatenate along the output dimension (dim 0 for linear weight)
        fused_weight = torch.cat([gate_weight, up_weight], dim=0)
        layer.w_gate_up.weight.data.copy_(fused_weight)

        if layer.w_gate_up.bias is not None:
            gate_bias = self.state.get(self.state.key(mlp_prefix, "gate_proj", "bias"))
            up_bias = self.state.get(self.state.key(mlp_prefix, "up_proj", "bias"))
            fused_bias = torch.cat([gate_bias, up_bias], dim=0)
            layer.w_gate_up.bias.data.copy_(fused_bias)

        # Load down projection
        layer.w_down.weight.data.copy_(
            self.state.get(self.state.key(mlp_prefix, "down_proj", "weight"))
        )
        if layer.w_down.bias is not None:
            layer.w_down.bias.data.copy_(
                self.state.get(self.state.key(mlp_prefix, "down_proj", "bias"))
            )

    def find_embedder(self) -> Embedder:
        """Find the model's embedder module."""
        for _, m in self.model.named_modules():
            if isinstance(m, Embedder):
                return m

        raise ValueError("No embedder found in model")

    def find_head(self) -> LinearLayer:
        """Find the model's LM head (last LinearLayer)."""
        heads = [m for _, m in self.model.named_modules() if isinstance(m, LinearLayer)]
        if not heads:
            raise ValueError("No LM head found in model")
        return heads[-1]
