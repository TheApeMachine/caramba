"""The Model class: composing embeddings, transformer layers, and output heads.

A language model is more than just transformer blocks—it needs an embedder to
convert token IDs to vectors, a stack of transformer layers to process them,
and an output projection to produce vocabulary logits. This module ties all
those pieces together based on a ModelConfig.
"""
from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from console import logger
from config.diffusion import DiffusionHeadConfig
from config.model import ModelConfig
from layer.diffusion_head import (
    DIFFUSERS_AVAILABLE,
    DiffusionHeadConfig as RuntimeDiffusionConfig,
    DiffusionNextTokenHead,
)
from model.embedder import Embedder

RuntimeDiffusionConfigType = RuntimeDiffusionConfig


class Model(nn.Module):
    """A complete language model: embeddings → transformer → output logits.

    The Model class is the top-level container that wires together:
    - An embedder (token IDs → vectors)
    - A topology (the transformer layer stack)
    - Optionally, a diffusion head for hybrid generation

    For upcycling, we create two Models with identical configs except for
    the attention type—one standard (teacher) and one DBA (student).
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build a model from configuration.

        The config specifies the embedder type, transformer topology, and
        whether to include a diffusion head for hybrid token generation.
        """
        super().__init__()
        self.config = config
        self.embedder = Embedder(config.embedder)
        self.topology = config.topology.build()

        # Optional diffusion head for denoising-based generation
        self.diffusion_head: DiffusionNextTokenHead | None = None
        if config.diffusion_head.enabled:
            if not DIFFUSERS_AVAILABLE:
                raise RuntimeError(
                    "diffusion_head.enabled=True but `diffusers` is not installed. "
                    "Install with: pip install diffusers"
                )
            runtime_cfg = config.diffusion_head.to_runtime_config()
            if not isinstance(runtime_cfg, RuntimeDiffusionConfig):
                raise TypeError(
                    f"Expected RuntimeDiffusionConfig, got {type(runtime_cfg).__name__}"
                )
            embed_dim = self._get_embed_dim()
            self.diffusion_head = DiffusionNextTokenHead(
                embed_dim=embed_dim,
                cfg=runtime_cfg,
            )

    def _get_embed_dim(self) -> int:
        """Extract the embedding dimension for the diffusion head."""
        from config.embedder import TokenEmbedderConfig

        if isinstance(self.config.embedder, TokenEmbedderConfig):
            return self.config.embedder.d_model
        raise ValueError(
            "Cannot determine embed_dim for diffusion head: "
            "embedder must be TokenEmbedderConfig"
        )

    @override
    def forward(
        self,
        x: Tensor,
        *,
        ctx: object | None = None,
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Run the full model: embed → transform → project to vocab.

        Args:
            x: Token IDs, shape (B, T)
            ctx: Optional inference context (e.g. KV-cache) passed through
                 to topology/layers during generation.
            return_features: If True, also return pre-logit hidden states
                            (needed for diffusion head training)

        Returns:
            Logits (B, T, vocab_size), or (features, logits) if return_features=True
        """
        x = self.embedder(x)
        features = self.topology(x, ctx=ctx)  # type: ignore[call-arg]

        if return_features:
            logits = self._features_to_logits(features)
            return features, logits

        return self._features_to_logits(features)

    def _features_to_logits(self, features: Tensor) -> Tensor:
        """Project hidden states to vocabulary logits.

        With tied embeddings, we reuse the embedding matrix as the output
        projection (features @ embedding.weight.T). Without tied embeddings,
        the topology already includes an LM head.
        """
        if not self.config.tied_embeddings:
            return features
        emb = self.embedder.token_embedding
        if emb is None:
            return features

        # Compatibility: some presets/topologies include an explicit LM head
        # (LinearLayer d_out=vocab_size) while also leaving tied_embeddings=True.
        # In that case, `features` are already logits and should not be projected again.
        try:
            vocab_size, d_model = emb.weight.shape
            last_dim = int(features.shape[-1])
        except Exception:
            return features

        if last_dim == int(vocab_size):
            return features
        if last_dim != int(d_model):
            raise ValueError(
                "Model output dimension mismatch for tied embeddings: "
                f"features.shape[-1]={last_dim} but token_embedding.weight.shape={(int(vocab_size), int(d_model))}. "
                "Either set tied_embeddings=false or adjust the topology output dimension."
            )

        return features @ emb.weight.t()

    def diffusion_loss(
        self,
        features: Tensor,
        target_ids: Tensor,
    ) -> Tensor:
        """Compute the diffusion denoising loss for hybrid training.

        The diffusion head learns to denoise target embeddings conditioned
        on transformer features. This provides an alternative generation
        path that can be more controllable than pure autoregressive sampling.
        """
        if self.diffusion_head is None:
            raise RuntimeError("diffusion_loss called but diffusion_head is not enabled")
        if self.embedder.token_embedding is None:
            raise RuntimeError("diffusion_loss requires token embeddings")

        target_emb = self.embedder.token_embedding(target_ids)
        return self.diffusion_head.diffusion_loss(cond=features, target_emb=target_emb)

    def sample_with_diffusion(
        self,
        features_last: Tensor,
        *,
        temperature: float = 1.0,
        guidance_scale: float | None = None,
    ) -> Tensor:
        """Sample next-token logits using the diffusion head.

        Instead of directly using the model's logit output, we run a
        denoising diffusion process conditioned on the last position's
        features to generate a clean embedding, then project to logits.
        """
        if self.diffusion_head is None:
            raise RuntimeError(
                "sample_with_diffusion called but diffusion_head is not enabled"
            )
        if self.embedder.token_embedding is None:
            raise RuntimeError("sample_with_diffusion requires token embeddings")

        return self.diffusion_head.sample_next_logits(
            cond_last=features_last,
            tok_emb_weight_t=self.embedder.token_embedding.weight.t(),
            temperature=temperature,
            guidance_scale=guidance_scale,
        )

    # HuggingFace-style helpers (used by benchmarks/introspection)
    def get_input_embeddings(self) -> nn.Module | None:
        """Return the token embedding module if present.

        Why this exists:
        - Some generic utilities (benchmarks, exports) expect the HF-style
          `get_input_embeddings()` API to determine vocab size.
        """
        return self.embedder.token_embedding

    @property
    def vocab_size(self) -> int | None:
        """Best-effort vocabulary size for token models."""
        emb = self.embedder.token_embedding
        if emb is None:
            logger.error("Failed to get vocab size, continuing: embedder.token_embedding is None")
            return None
        try:
            return int(emb.num_embeddings)
        except Exception as e:
            logger.error(f"Failed to get vocab size, continuing: {e}")
            return None
