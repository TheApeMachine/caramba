"""Apply rotary position embedding (RoPE)

Injects positional information through rotation of query/key vectors.
"""
from __future__ import annotations

import torch
from torch import Tensor

from caramba.operation.positional.base import PositionalOperation


class ApplyRoPEOperation(PositionalOperation):
    """Apply rotary position embedding

    Rotates query and key vectors to encode positional information,
    allowing attention to naturally capture relative distances between positions.
    """
    def __init__(
        self,
        base: float = 10000.0,
        variant: str = "both",  # "both", "q_only", "k_only"
    ) -> None:
        super().__init__()
        self.base = base
        self.variant = variant

    def _get_rotary_embedding(self, seq_len: int, dim: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Generate rotary position embeddings

        Creates cos/sin matrices for rotating vectors based on their positions.
        """
        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

        # Generate position indices
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)

        # Compute angles
        angles = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)

        # Create cos/sin embeddings
        cos = angles.cos()
        sin = angles.sin()

        return cos, sin

    def _apply_rotary_pos_emb(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply rotary position embedding to input tensor"""
        # Split input into even and odd dimensions
        x1, x2 = x[..., ::2], x[..., 1::2]

        # Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return rotated

    def forward(self, *, q: Tensor, k: Tensor, start_pos: int = 0) -> tuple[Tensor, Tensor]:
        """Apply RoPE to queries and keys

        Rotates query and key vectors to inject positional information,
        with options to apply only to queries, keys, or both.
        """
        start_pos = int(start_pos)
        bq, hq, tq, dq = q.shape
        bk, hk, tk, dk = k.shape
        if dq != dk:
            raise ValueError(f"q/k head_dim mismatch: {dq} != {dk}")
        if tq != tk:
            raise ValueError(f"q/k seq_len mismatch: {tq} != {tk}")

        # Get rotary embeddings for this sequence length.
        cos, sin = self._get_rotary_embedding(tq + start_pos, dq, q.device)

        # Extract the relevant portion for current sequence.
        cos = cos[start_pos:start_pos + tq]  # [seq_len, head_dim/2]
        sin = sin[start_pos:start_pos + tq]  # [seq_len, head_dim/2]
        # Match input dtype to avoid unwanted promotions (important for cache ops
        # and SDPA which expect q/k/v to share dtype).
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)

        # Expand per-input to avoid accidentally broadcasting heads (important for GQA).
        cos_q = cos.unsqueeze(0).unsqueeze(0).expand(int(bq), int(hq), -1, -1)
        sin_q = sin.unsqueeze(0).unsqueeze(0).expand(int(bq), int(hq), -1, -1)
        cos_k = cos.unsqueeze(0).unsqueeze(0).expand(int(bk), int(hk), -1, -1)
        sin_k = sin.unsqueeze(0).unsqueeze(0).expand(int(bk), int(hk), -1, -1)

        # Apply to queries if requested
        q_out = q
        if self.variant in ["both", "q_only"]:
            q_out = self._apply_rotary_pos_emb(q, cos_q, sin_q)

        # Apply to keys if requested
        k_out = k
        if self.variant in ["both", "k_only"]:
            k_out = self._apply_rotary_pos_emb(k, cos_k, sin_k)

        return q_out, k_out
