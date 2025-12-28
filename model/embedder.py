"""Token embeddings: converting discrete token IDs to continuous vectors.

Language models operate on continuous vectors, but text is discrete tokens.
The embedder bridges this gap by mapping each token ID to a learned vector.
These embeddings encode semantic meaning—similar tokens end up with similar
vectors after training.
"""
from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from config.embedder import EmbedderConfig, EmbedderType


class Embedder(nn.Module):
    """Converts token IDs to embedding vectors.

    For language models, this is typically a learned embedding table where
    each of the vocab_size tokens has its own d_model-dimensional vector.
    The "none" type is for models that receive pre-embedded inputs.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        """Initialize the embedding layer based on config.

        TOKEN type creates a learnable embedding table. NONE type passes
        inputs through unchanged (for pre-embedded data). PATCH type
        uses a 2D convolution to embed image patches.
        """
        super().__init__()
        self.config = config
        self.token_embedding: nn.Embedding | None = None
        self.patch_embedding: nn.Conv2d | None = None

        match config.type:
            case EmbedderType.NONE:
                self.token_embedding = None
            case EmbedderType.TOKEN:
                self.token_embedding = nn.Embedding(
                    num_embeddings=config.vocab_size,
                    embedding_dim=config.d_model,
                )
            case EmbedderType.PATCH:
                self.patch_embedding = nn.Conv2d(
                    in_channels=config.in_channels,
                    out_channels=config.d_model,
                    kernel_size=config.patch_size,
                    stride=config.patch_size,
                )
            case _:
                raise ValueError(f"Unknown embedder type: {config.type}")

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Look up embeddings for input token IDs or images.

        Args:
            x: Token IDs (B, T) or image tensor (B, C, H, W)

        Returns:
            Embeddings, shape (B, T, d_model)
        """
        match self.config.type:
            case EmbedderType.NONE:
                return x
            case EmbedderType.TOKEN:
                if self.token_embedding is None:
                    raise RuntimeError("Token embedder is not initialized.")
                return self.token_embedding(
                    x.to(dtype=self.token_embedding.weight.dtype).long(),
                )
            case EmbedderType.PATCH:
                if self.patch_embedding is None:
                    raise RuntimeError("Patch embedder is not initialized.")
                # x: (B, C, H, W) -> (B, d_model, H/P, W/P)
                x = self.patch_embedding(x)
                # -> (B, d_model, T) where T = (H/P * W/P)
                x = x.flatten(2)
                # -> (B, T, d_model)
                return x.transpose(1, 2)
            case _:
                raise ValueError(f"Unknown embedder type: {self.config.type}")
