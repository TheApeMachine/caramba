"""Layer primitives

This package is the model's vocabulary of building blocks: small, composable
`nn.Module`s that can be assembled by manifests into many different
architectures without rewriting model code.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from carmath import neg_inf
from config.layer import LayerConfig


class Layer(nn.Module):
    """Layer base class

    A shared interface makes layers interchangeable inside manifest-driven
    topologies, so you can swap implementations (attention, MLP, norms) without
    changing the orchestration code that wires a model together.
    """
    def __init__(self, config: LayerConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass

        Layers take tensors in, tensors out; the point of the base class is to
        standardize the signature so composition stays frictionless.
        """
        raise NotImplementedError("Subclasses must implement forward pass.")

    def cross_attention(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Cross-attention helper

        This is the “scaled dot-product attention” core, extracted as a small
        utility so non-attention layers can reuse the math without importing a
        larger attention stack.
        """
        d_k = Q.size(-1)
        scale = math.sqrt(float(d_k))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, neg_inf(scores.dtype))

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
