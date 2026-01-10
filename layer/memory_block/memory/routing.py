"""Memory routing

Computes constant-time bucket indices for memory reads/writes.
Routing is driven by the **tag/key embedding** (not the full hidden state) to
improve invariance under paraphrase and distractors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class Routing:
    """Routing outputs

    Routing turns continuous tags into discrete bucket indices, which is what
    makes the memory operations constant-time with respect to total memory size.
    """

    idx_r: Tensor
    idx_w: Tensor
    aux: dict[str, Tensor]


class BitRouter(nn.Module):
    """Sign-bit router.

    Produces bucket indices via learned projections to sign bits.
    """

    def __init__(self, *, in_dim: int, hashes: int, buckets: int) -> None:
        super().__init__()
        self.hashes = int(hashes)
        self.buckets = int(buckets)
        self.bits = int((self.buckets - 1).bit_length())
        self.read = nn.Linear(int(in_dim), int(hashes) * int(self.bits), bias=False)
        self.write = nn.Linear(int(in_dim), int(hashes) * int(self.bits), bias=False)
        self.bit_powers_cache: Tensor | None = None

    def bit_powers(self, device: torch.device) -> Tensor:
        p = self.bit_powers_cache
        if p is None or p.device != device or int(p.numel()) != int(self.bits):
            p = (2 ** torch.arange(self.bits, device=device, dtype=torch.long)).view(1, 1, self.bits)
            self.bit_powers_cache = p
        return p

    def route(self, *, tag: Tensor, collect_aux: bool) -> Routing:
        if not isinstance(tag, Tensor):
            raise TypeError(f"tag must be a Tensor, got {type(tag).__name__}")
        if tag.ndim != 3:
            raise ValueError(f"tag must have shape (B,T,D), got {tuple(tag.shape)}")
        B, T, _ = tag.shape
        z_r = self.read(tag).view(B, T, self.hashes, self.bits)
        z_w = self.write(tag).view(B, T, self.hashes, self.bits)
        idx_r = self.bits_to_idx(z_r)
        idx_w = self.bits_to_idx(z_w)
        aux: dict[str, Tensor] = {}
        if collect_aux:
            aux["read_bit_logits"] = z_r
            aux["write_bit_logits"] = z_w
        return Routing(idx_r=idx_r, idx_w=idx_w, aux=aux)

    def bits_to_idx(self, z: Tensor) -> Tensor:
        bits = z > 0
        idx = (bits.to(torch.long) * self.bit_powers(z.device)).sum(dim=-1)
        if self.is_power_of_two(self.buckets) and (1 << self.bits) == self.buckets:
            return idx
        return torch.remainder(idx, int(self.buckets))

    def is_power_of_two(self, n: int) -> bool:
        v = int(n)
        return v > 0 and (v & (v - 1)) == 0


class VqRouter(nn.Module):
    """Product-quantized router

    VQ routing is a learned, discrete indexing mechanism: it maps tags to
    codebook entries per group and combines them into bucket ids, which can be
    more stable than raw sign-bit hashing for some distributions.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        hashes: int,
        buckets: int,
        groups: int,
        codebook_size: int,
        group_dim: int,
        beam: int,
        write_multi: bool,
    ) -> None:
        super().__init__()
        self.hashes = int(hashes)
        self.buckets = int(buckets)
        self.groups = int(groups)
        self.codebook_size = int(codebook_size)
        self.group_dim = int(group_dim)
        self.beam = int(beam)
        self.write_multi = bool(write_multi)
        route_dim = int(self.groups * self.group_dim)
        self.proj_r = nn.Linear(int(in_dim), int(hashes) * route_dim, bias=False)
        self.proj_w = nn.Linear(int(in_dim), int(hashes) * route_dim, bias=False)
        self.codebook_r = nn.Parameter(torch.randn(self.hashes, self.groups, self.codebook_size, self.group_dim) * 0.02)
        self.codebook_w = nn.Parameter(torch.randn(self.hashes, self.groups, self.codebook_size, self.group_dim) * 0.02)
        self.powK_cache: Tensor | None = None
        self.combo_cache: Tensor | None = None

    def route(self, *, tag: Tensor, collect_aux: bool) -> Routing:
        if tag.ndim != 3:
            raise ValueError(f"tag must have shape (B,T,D), got {tuple(tag.shape)}")
        B, T, _ = tag.shape
        route_dim = int(self.groups * self.group_dim)
        y_r = self.proj_r(tag).view(B, T, self.hashes, route_dim)
        y_w = self.proj_w(tag).view(B, T, self.hashes, route_dim)
        idx_r, aux_r = self.route_side(y=y_r, codebook=self.codebook_r)
        idx_w, aux_w = self.route_side(y=y_w, codebook=self.codebook_w)
        aux: dict[str, Tensor] = {}
        if collect_aux:
            aux["read_vq_logits"] = aux_r
            aux["write_vq_logits"] = aux_w
        return Routing(idx_r=idx_r, idx_w=idx_w, aux=aux)

    def route_side(self, *, y: Tensor, codebook: Tensor) -> tuple[Tensor, Tensor]:
        B, T, H, _ = y.shape
        y_g = y.view(B, T, H, self.groups, self.group_dim)
        cb = codebook.view(1, 1, H, self.groups, self.codebook_size, self.group_dim)
        diff = y_g.unsqueeze(-2) - cb
        dist = (diff * diff).sum(dim=-1)  # (B,T,H,G,K)
        logits = (-dist).to(dtype=y.dtype)
        if self.beam <= 1:
            codes = logits.argmax(dim=-1)  # (B,T,H,G)
            idx = self.codes_to_bucket(codes)
            return idx, logits
        top = torch.topk(logits, k=int(self.beam), dim=-1).indices  # (B,T,H,G,beam)
        combo = self.combo_index(device=y.device)  # (Nc,G)
        cand = torch.gather(top, dim=-1, index=combo.view(1, 1, 1, self.groups, -1).expand(B, T, H, self.groups, -1))
        codes = cand.permute(0, 1, 2, 4, 3)  # (B,T,H,Nc,G)
        idx = self.codes_to_bucket(codes)  # (B,T,H,Nc)
        idx0 = idx[..., 0]
        return idx0, logits

    def codes_to_bucket(self, codes: Tensor) -> Tensor:
        powK = self.powK(device=codes.device)
        if codes.ndim == 4:
            v = (codes.to(torch.long) * powK.view(1, 1, 1, self.groups)).sum(dim=-1)
            return torch.remainder(v, int(self.buckets))
        if codes.ndim == 5:
            v = (codes.to(torch.long) * powK.view(1, 1, 1, 1, self.groups)).sum(dim=-1)
            return torch.remainder(v, int(self.buckets))
        raise ValueError(f"Unsupported codes shape {tuple(codes.shape)}")

    def powK(self, device: torch.device) -> Tensor:
        p = self.powK_cache
        if p is None or p.device != device or int(p.numel()) != int(self.groups):
            p = torch.tensor([int(self.codebook_size) ** i for i in range(int(self.groups))], device=device, dtype=torch.long)
            self.powK_cache = p
        return p

    def combo_index(self, device: torch.device) -> Tensor:
        if self.beam <= 1:
            return torch.zeros((1, int(self.groups)), device=device, dtype=torch.long)
        cached = self.combo_cache
        if cached is not None and cached.device == device and int(cached.shape[1]) == int(self.groups):
            return cached
        grids = torch.cartesian_prod(*([torch.arange(int(self.beam), device=device)] * int(self.groups)))
        self.combo_cache = grids.to(dtype=torch.long)
        return self.combo_cache


class ResonantRouter(nn.Module):
    """Resonant iterative router

    Ported from resonant.core.associative_memory. Uses phase dynamics
    to find the best-matching bucket (attractor).
    """

    def __init__(
        self,
        *,
        in_dim: int,
        hashes: int,
        buckets: int,
        steps: int = 20,
        coupling: float = 0.25,
        damping: float = 0.02,
        zero_diag: bool = True,
        tuner: Any | None = None,
    ) -> None:
        super().__init__()
        self.hashes = int(hashes)
        self.buckets = int(buckets)
        self.steps = int(steps)
        self.coupling = float(coupling)
        self.damping = float(damping)
        self.num_units = int(in_dim)
        self.zero_diag = bool(zero_diag)
        self.tuner = tuner

        # patterns are our "codebook" but in phase space
        self.patterns = nn.Parameter(torch.randn(self.hashes, self.buckets, self.num_units) * 0.02)

    def _check_nan(self, x: Tensor, name: str) -> None:
        if not torch.isfinite(x).all():
            print(f"!!! [ResonantRouter] {name} is NOT FINITE (NaN/Inf) !!!")

    def route(self, *, tag: Tensor, collect_aux: bool) -> Routing:
        # tag: (B, T, D)
        if tag.ndim != 3:
            raise ValueError(f"tag must have shape (B,T,D), got {tuple(tag.shape)}")
        B, T, D = tag.shape
        H = self.hashes
        device = tag.device

        # Get current parameters (potentially tuned)
        steps = self.steps
        coupling = self.coupling
        damping = self.damping
        
        if self.tuner is not None:
            # Apply scaling factors from tuner (tuner.resonant_coupling_mult etc.)
            coupling = coupling * getattr(self.tuner, "resonant_coupling_mult", 1.0)
            damping = damping * getattr(self.tuner, "resonant_damping_mult", 1.0)
            steps = steps + getattr(self.tuner, "resonant_steps_delta", 0)

        # Safety clamp to prevent instability
        tag = torch.nan_to_num(tag, nan=0.0, posinf=100.0, neginf=-100.0)
        tag = tag.clamp(-100.0, 100.0)
        self._check_nan(tag, "tag (input)")

        # 1. Project Patterns to Real/Imag parts (A, B)
        # patterns is (H, K, D)
        p_phases = self.patterns.clamp(-100.0, 100.0)
        A = torch.cos(p_phases)  # Real: (H, K, D)
        B_ = torch.sin(p_phases) # Imag: (H, K, D)

        # 2. Project Tag to Real/Imag parts (x, y)
        x = torch.cos(tag).unsqueeze(2).expand(B, T, H, D).clone() # (B, T, H, D)
        y = torch.sin(tag).unsqueeze(2).expand(B, T, H, D).clone() # (B, T, H, D)
        
        # 3. Precompute Real-valued Hebbian Weights (W_r, W_i)
        At = A.transpose(-1, -2) # (H, D, K)
        Bt = B_.transpose(-1, -2) # (H, D, K)
        
        Wr = (torch.matmul(At, A) + torch.matmul(Bt, B_)) / float(D) # (H, D, D)
        Wi = (torch.matmul(At, B_) - torch.matmul(Bt, A)) / float(D) # (H, D, D)
        
        if self.zero_diag:
            mask = torch.eye(D, device=device, dtype=torch.bool).unsqueeze(0).expand(H, D, D)
            Wr = Wr.masked_fill(mask, 0.0)
            Wi = Wi.masked_fill(mask, 0.0)

        # 4. Iterative Dynamics (Settling) using Real-Valued Components
        scale = coupling / float(self.buckets)
        
        # For Telemetry: track energy and convergence
        energy_history = []
        
        for s_idx in range(int(steps)):
            xr = x.view(-1, H, 1, D)
            yi = y.view(-1, H, 1, D)
            
            c_r = (torch.matmul(xr, Wr) - torch.matmul(yi, Wi)).view(B, T, H, D)
            c_i = (torch.matmul(xr, Wi) + torch.matmul(yi, Wr)).view(B, T, H, D)
            
            x = (x + scale * c_r) * (1.0 - damping)
            y = (y + scale * c_i) * (1.0 - damping)

            mag = torch.sqrt(x**2 + y**2 + 1e-12)
            x = x / mag
            y = y / mag
            
            if collect_aux and s_idx % 5 == 0:
                # Approximate energy (squared similarity sum)
                energy = (x**2 + y**2).mean().item()
                energy_history.append(energy)

        # 5. Final Similarity calculation using real components
        xr = x.unsqueeze(-2)
        yi = y.unsqueeze(-2)
        At = A.transpose(-1, -2)
        Bt = B_.transpose(-1, -2)
        
        dot_r = (torch.matmul(xr, At) + torch.matmul(yi, Bt)).squeeze(-2) # (B, T, H, K)
        dot_i = (torch.matmul(yi, At) - torch.matmul(xr, Bt)).squeeze(-2) # (B, T, H, K)
        
        sim = torch.sqrt(dot_r**2 + dot_i**2 + 1e-12) / float(D)
        sim = sim.clamp(0.0, 1.0)
        
        logits = sim
        idx_r = logits.argmax(dim=-1)
        idx_w = idx_r

        aux: dict[str, Tensor] = {}
        if collect_aux:
            aux["read_resonant_logits"] = logits
            aux["write_resonant_logits"] = logits
            
            # Telemetry: final max similarity
            max_sim = logits.max(dim=-1)[0].mean().item()
            aux["resonant_final_sim"] = torch.tensor(max_sim, device=device)
            
            # Telemetry: Entropy of routing (how diversified are the buckets?)
            if B * T > 1:
                flat_idx = idx_r.view(-1)
                counts = torch.bincount(flat_idx, minlength=self.buckets).float()
                probs = counts / counts.sum().clamp_min(1e-9)
                entropy = -(probs * torch.log(probs + 1e-9)).sum().item()
                aux["resonant_bucket_entropy"] = torch.tensor(entropy, device=device)

        return Routing(idx_r=idx_r, idx_w=idx_w, aux=aux)

