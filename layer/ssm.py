"""State Space Model (SSM) layer implementation.

SSMs process sequences by maintaining a hidden state that evolves over time.
Selective SSMs (like Mamba) make this evolution data-dependent, allowing the
model to focus on relevant information and achieve linear scaling with
sequence length.

Our implementation uses a parallel associative scan to process sequences in
O(log T) time, ensuring that caramba provides a foundation for state-of-the-art
SSM research.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from caramba.config.layer import SSMLayerConfig


class _SelectiveScan:
    """Selective scan implementation

    A simple affine recurrence becomes parallelizable when you view each timestep
    as an associative “transform”, which is why SSMs can be both recurrent in
    spirit and parallel in execution.
    """

    @staticmethod
    def _inclusive_affine_scan(a: Tensor, u: Tensor) -> Tensor:
        """Inclusive scan for an affine recurrence

        Scans trade a sequential loop for a small number of large tensor ops,
        which is often a win on accelerators even if total FLOPs increase.
        """
        if a.shape != u.shape:
            raise ValueError(f"scan expects a/u same shape, got a={tuple(a.shape)} u={tuple(u.shape)}")
        if a.ndim != 4:
            raise ValueError(f"scan expects a/u with 4 dims (B,T,D_inner,D_state), got {tuple(a.shape)}")
        T = int(a.shape[1])
        if T <= 0:
            raise ValueError("scan expects T > 0")

        # Hillis–Steele inclusive scan. Cost: O(T log T) work but only O(log T)
        # Python iterations; each iteration is a few large MPS-friendly tensor ops.
        #
        # Invariant: (a,u) at position t represents the composed transform from
        # time 0..t applied to h_{-1}=0, so u == h_t.
        k = 1
        while k < T:
            a_shift = torch.ones_like(a)
            u_shift = torch.zeros_like(u)
            a_shift[:, k:] = a[:, :-k]
            u_shift[:, k:] = u[:, :-k]
            u = u + a * u_shift
            a = a * a_shift
            k <<= 1
        return u

    @classmethod
    def selective_scan(
        cls,
        *,
        x: Tensor,
        dt: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
    ) -> Tensor:
        """Vectorized selective scan

        This is the core SSM computation: evolve a per-feature state through time
        and read it out with a learned projection to produce sequence features.
        """
        if x.ndim != 3:
            raise ValueError(f"expected x as (B,T,D_inner), got {tuple(x.shape)}")
        if dt.shape != x.shape:
            raise ValueError(f"expected dt shape == x shape, got dt={tuple(dt.shape)} x={tuple(x.shape)}")
        if A.ndim != 2:
            raise ValueError(f"expected A as (D_inner,D_state), got {tuple(A.shape)}")
        if B.ndim != 3 or C.ndim != 3:
            raise ValueError(f"expected B,C as (B,T,D_state), got B={tuple(B.shape)} C={tuple(C.shape)}")
        if D.ndim != 1:
            raise ValueError(f"expected D as (D_inner,), got {tuple(D.shape)}")

        B_size, T, D_inner = x.shape
        D_state = int(A.shape[1])
        if int(A.shape[0]) != int(D_inner):
            raise ValueError(f"A shape mismatch: expected A.shape[0]==D_inner ({D_inner}), got {int(A.shape[0])}")
        if B.shape[0] != B_size or B.shape[1] != T or int(B.shape[2]) != D_state:
            raise ValueError("B shape mismatch")
        if C.shape[0] != B_size or C.shape[1] != T or int(C.shape[2]) != D_state:
            raise ValueError("C shape mismatch")
        if int(D.numel()) != int(D_inner):
            raise ValueError(f"D shape mismatch: expected {D_inner}, got {int(D.numel())}")

        # Discretize.
        a = torch.exp(dt.unsqueeze(-1) * A.view(1, 1, D_inner, D_state))  # (B,T,D_inner,D_state)
        u = (dt.unsqueeze(-1) * B.unsqueeze(2)) * x.unsqueeze(-1)  # (B,T,D_inner,D_state)

        h = cls._inclusive_affine_scan(a, u)  # (B,T,D_inner,D_state)
        y = (h * C.unsqueeze(2)).sum(dim=-1)  # (B,T,D_inner)
        return y + x * D.view(1, 1, -1)


class SSMLayer(nn.Module):
    """Selective State Space Model layer (Mamba-style) with parallel scan.

    Combines a selective scan with a 1D convolution and gating. The scan is
    implemented using a parallel associative operator to maximize GPU throughput
    on long sequences.
    """

    def __init__(self, config: SSMLayerConfig) -> None:
        """Initialize SSM parameters

        The architecture mixes three ideas: local convolution for short-range
        patterns, a selective scan for long-range memory, and a gate to control
        information flow between them.
        """
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_inner = config.d_model * config.expand
        self.d_conv = config.d_conv

        # Input projection: splits into SSM path and gate path
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=config.bias)

        # 1D Convolution: local feature extraction
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=config.conv_bias,
            kernel_size=config.d_conv,
            groups=self.d_inner,
            padding=config.d_conv - 1,
        )

        # Selective projections: data-dependent B, C, and delta
        dt_rank = (
            math.ceil(self.d_model / 16)
            if config.dt_rank == "auto"
            else int(config.dt_rank)
        )
        self.x_proj = nn.Linear(self.d_inner, dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # SSM Parameters: A and D (learned but not data-dependent)
        # Initialize A using S4D-style (S4 with diagonal A)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(
            self.d_inner, 1
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)

    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        """Process a sequence with an SSM

        SSMs are “stateful” models, but with the scan trick they can be trained
        and run with parallel tensor ops instead of an explicit Python loop.
        """
        B, T, D = x.shape

        # 1. Input projection and split
        xz = self.in_proj(x)  # (B, T, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, T, d_inner) each

        # 2. Convolution path
        x = x.transpose(1, 2)  # (B, d_inner, T)
        x = self.conv1d(x)[:, :, :T]  # (B, d_inner, T)
        x = F.silu(x)
        x = x.transpose(1, 2)  # (B, T, d_inner)

        # 3. Selective Scan
        # Keep A in the same dtype as activations for fused kernels.
        A = (-torch.exp(self.A_log)).to(dtype=x.dtype)  # (d_inner, d_state)

        # Selective projections
        x_dbl = self.x_proj(x)  # (B, T, dt_rank + 2*d_state)
        dt, B_vals, C = torch.split(
            x_dbl, [self.x_proj.out_features - 2 * self.d_state, self.d_state, self.d_state], dim=-1
        )

        # dt: (B, T, d_inner)
        dt = F.softplus(self.dt_proj(dt))

        # Kernel contract: dt/B/C/D must match activation dtype.
        dt = dt.to(dtype=x.dtype)
        B_vals = B_vals.to(dtype=x.dtype)
        C = C.to(dtype=x.dtype)
        D_skip = self.D.to(dtype=x.dtype, device=x.device)

        # Selective Scan Path (best available kernel)
        if x.device.type == "mps" and x.dtype == torch.float16:
            from caramba.optimizer.metal import MetalSSMSelectiveScan

            scan = MetalSSMSelectiveScan()
            y = scan.run(x=x, dt=dt, A=A, B=B_vals, C=C, D=D_skip)
        elif x.device.type == "cuda":
            from caramba.optimizer.fused_ssm import fused_selective_scan

            y = fused_selective_scan(x, dt, A, B_vals, C, D_skip)
        else:
            y = self._selective_scan_parallel(x, dt, A, B_vals, C, D_skip)

        # 4. Gating and output projection
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan_parallel(
        self, x: Tensor, dt: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor
    ) -> Tensor:
        """Selective scan using an associative scan (O(log T) steps)."""
        return _SelectiveScan.selective_scan(x=x, dt=dt, A=A, B=B, C=C, D=D)
