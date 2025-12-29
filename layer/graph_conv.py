"""High-performance Graph Convolution layers.

Implements GCN (Graph Convolutional Network) and GAT (Graph Attention Network)
using sparse matrix operations for efficiency without external dependencies
like torch_geometric.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from typing_extensions import override

from config.layer import GraphConvLayerConfig

if TYPE_CHECKING:
    from tensordict import TensorDictBase


class GraphConvLayer(nn.Module):
    """Graph Convolution Layer (GCN or GAT).

    Expects input to be a tuple (x, adj) where:
    - x: Node features (N, in_features)
    - adj: Adjacency matrix (N, N) - can be sparse or dense
           OR edge_index (2, E)
    """

    def __init__(self, config: GraphConvLayerConfig) -> None:
        super().__init__()
        self.config = config

        if config.kind == "gcn":
            self.impl = _GCNImpl(config)
        elif config.kind == "gat":
            self.impl = _GATImpl(config)
        else:
            raise ValueError(f"Unknown graph layer kind: {config.kind}")

    @override
    def forward(
        self,
        x: Tensor | tuple[Tensor, Tensor],
        *,
        ctx: object | None = None,
    ) -> Tensor:
        """Apply graph convolution.

        Args:
            x: Either a Tensor of features (if adj is in ctx) or tuple (features, adj)
        """
        adj: Tensor | None = None

        if isinstance(x, tuple):
            feat, adj = x
        else:
            feat = x
            # Try to find adj in context if not explicit
            if ctx is not None and hasattr(ctx, "adj"):
                adj = getattr(ctx, "adj")

        if adj is None:
            raise ValueError("GraphConvLayer requires adjacency matrix/edge_index in input tuple or context")

        return self.impl(feat, adj)


class _GCNImpl(nn.Module):
    """Kipf & Welling GCN implementation."""

    def __init__(self, config: GraphConvLayerConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.in_features, config.out_features, bias=config.bias)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        # X' = A * X * W
        # 1. Linear projection: X_w = X * W
        x_w = self.linear(x)

        # 2. Sparse/Dense Matmul: A * X_w
        if adj.is_sparse:
            out = torch.sparse.mm(adj, x_w)
        else:
            out = torch.matmul(adj, x_w)

        return out


class _GATImpl(nn.Module):
    """Graph Attention Network (GAT) implementation.

    Uses dense implementation for simplicity and batching support,
    optimized for smaller graphs or node-classification benchmarks.
    """

    def __init__(self, config: GraphConvLayerConfig) -> None:
        super().__init__()
        self.in_features = config.in_features
        self.out_features = config.out_features
        self.heads = config.heads
        self.concat = config.concat
        self.dropout_p = config.dropout
        self.negative_slope = config.negative_slope

        self.head_dim = config.out_features // config.heads if config.concat else config.out_features

        self.W = nn.Linear(config.in_features, self.heads * self.head_dim, bias=False)
        self.a_src = Parameter(torch.zeros(1, self.heads, self.head_dim))
        self.a_dst = Parameter(torch.zeros(1, self.heads, self.head_dim))

        if config.bias:
            self.bias = Parameter(torch.zeros(config.out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        N, C = x.shape

        # 1. Linear Projection
        # (N, heads * head_dim) -> (N, heads, head_dim)
        h = self.W(x).view(N, self.heads, self.head_dim)

        # 2. Attention mechanism
        # attn_src: (N, heads, 1)
        # attn_dst: (N, heads, 1)
        attn_src = (h * self.a_src).sum(dim=-1, keepdim=True)
        attn_dst = (h * self.a_dst).sum(dim=-1, keepdim=True)

        # attn: (N, N, heads) = src + dst.T
        # We broadcast to form the N x N attention matrix
        # scores = attn_src + attn_dst.transpose(0, 1)  # Logic needs careful broadcasting
        # Actually: (N, heads, 1) + (1, heads, N) -> (N, heads, N) -> permute to (N, N, heads)
        scores = attn_src.permute(1, 0, 2) + attn_dst.permute(1, 2, 0) # (heads, N, N)
        scores = F.leaky_relu(scores, self.negative_slope)

        # Mask with adjacency
        # If adj is dense (N, N)
        if adj.dim() == 2:
            # Broadcast adj to heads: (heads, N, N)
            mask = adj.unsqueeze(0)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout_p, training=self.training)

        # 3. Aggregation
        # (heads, N, N) @ (heads, N, head_dim) -> (heads, N, head_dim)
        # We need h in (heads, N, head_dim)
        h_prime = h.permute(1, 0, 2)
        out = torch.matmul(attn, h_prime) # (heads, N, head_dim)

        # 4. Concatenation / Averaging
        if self.concat:
            # (N, heads * head_dim)
            out = out.permute(1, 0, 2).reshape(N, self.heads * self.head_dim)
        else:
            # (N, head_dim)
            out = out.mean(dim=0)

        if self.bias is not None:
            out = out + self.bias

        return out
