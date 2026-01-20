"""Memory routing

Computes constant-time bucket indices for memory reads/writes.
Routing is driven by the **tag/key embedding** (not the full hidden state) to
improve invariance under paraphrase and distractors.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from optimizer.metal.resonant_update import MetalResonantPhaseUpdate
from optimizer.resonant_update_triton import ResonantPhaseUpdateTriton

_logger = logging.getLogger(__name__)


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
        tuner_mode: str = "off",
    ) -> None:
        super().__init__()
        self.hashes = int(hashes)
        self.buckets = int(buckets)
        self.steps = int(steps)
        self.coupling = float(coupling)
        self.damping = float(damping)
        self.num_units = int(in_dim)
        self.zero_diag = bool(zero_diag)
        self.tuner_mode = tuner_mode

        # patterns are our "codebook" but in phase space
        self.patterns = nn.Parameter(torch.randn(self.hashes, self.buckets, self.num_units) * 0.02)
        # Perf: cache diagonal mask per-device (avoid realloc each forward).
        self._zero_diag_mask_cache: dict[str, Tensor] = {}
        # Perf: cache derived tensors from patterns (fp32) keyed by (device, patterns_version).
        # This eliminates recomputing A/B and related projections when patterns are unchanged.
        self._patterns_cache: dict[str, tuple[int, Tensor, Tensor, Tensor, Tensor, Tensor]] = {}
        # Fused pointwise update kernels (max performance on accelerators).
        self._update_cuda: ResonantPhaseUpdateTriton | None = ResonantPhaseUpdateTriton() if torch.cuda.is_available() else None
        self._update_mps: MetalResonantPhaseUpdate | None = MetalResonantPhaseUpdate() if torch.backends.mps.is_available() else None

    def _check_nan(self, x: Tensor, name: str) -> None:
        if not torch.isfinite(x).all():
            non_finite = int((~torch.isfinite(x)).sum().item())
            _logger.warning(
                "ResonantRouter: %s has non-finite values (non_finite=%s shape=%s dtype=%s)",
                name,
                non_finite,
                tuple(x.shape),
                x.dtype,
            )

    def _derived_from_patterns(self, *, device: torch.device, D: int, H: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Derived tensors for resonant routing.

        Computes:
        - A, B_: (H, K, D) real/imag patterns
        - At, Bt: (H, D, K) transposes
        - diag: (H, D) diagonal of W for zero-diag correction
        """
        if self.patterns.ndim != 3:
            raise ValueError(f"patterns must have shape (H,K,D), got {tuple(self.patterns.shape)}")
        if int(self.patterns.shape[0]) != int(H) or int(self.patterns.shape[2]) != int(D):
            raise ValueError(
                f"patterns shape mismatch: expected H={H} D={D}, got {tuple(self.patterns.shape)}"
            )
        ver = int(getattr(self.patterns, "_version", 0))
        key = f"{device.type}:{device.index}:{int(H)}:{int(D)}"
        cached = self._patterns_cache.get(key, None)
        if cached is not None and int(cached[0]) == ver:
            # (ver, A, B, At, Bt, diag)
            _, A, B_, At, Bt, diag = cached
            if A.device == device:
                return A, B_, At, Bt, diag

        p_phases = self.patterns.to(device=device, dtype=torch.float32).clamp(-100.0, 100.0)
        A = torch.cos(p_phases)
        B_ = torch.sin(p_phases)
        At = A.transpose(-1, -2).contiguous()
        Bt = B_.transpose(-1, -2).contiguous()
        # diag(W) where W = P^H P / D and P = A + iB: diag = sum_k |P_k,i|^2 / D
        diag = ((A * A + B_ * B_).sum(dim=1) / float(D)).contiguous()  # (H,D)

        self._patterns_cache[key] = (ver, A, B_, At, Bt, diag)
        return A, B_, At, Bt, diag

    def route(self, *, tag: Tensor, collect_aux: bool) -> Routing:
        # tag: (B, T, D)
        if tag.ndim != 3:
            raise ValueError(f"tag must have shape (B,T,D), got {tuple(tag.shape)}")
        B, T, D = tag.shape
        H = self.hashes
        device = tag.device

        # Get current parameters (potentially tuned)
        base_steps = self.steps
        steps = base_steps
        coupling = self.coupling
        damping = self.damping

        if self.tuner_mode != "off":
            from layer.memory_block.memory.tuner import get_shared_tuner
            tuner = get_shared_tuner(mode=self.tuner_mode)

            # Apply scaling factors from tuner (tuner.resonant_coupling_mult etc.)
            coupling = coupling * getattr(tuner, "resonant_coupling_mult", 1.0)
            damping = damping * getattr(tuner, "resonant_damping_mult", 1.0)
            steps = steps + getattr(tuner, "resonant_steps_delta", 0)
            # Hard runtime guard: never allow tuning to explode step count.
            # This keeps autotune from causing multi-hour regressions.
            steps = int(max(1, min(int(steps), int(base_steps) + 5)))

        # Safety clamp to prevent instability.
        # IMPORTANT: run resonant routing math in fp32 for numerical stability,
        # even when the broader model runs in fp16. This targets precision where
        # it matters (attractor settling) without requiring fp32 everywhere.
        tag_f = tag.to(dtype=torch.float32)
        tag_f = torch.nan_to_num(tag_f, nan=0.0, posinf=100.0, neginf=-100.0)
        tag_f = tag_f.clamp(-100.0, 100.0)
        self._check_nan(tag_f, "tag (input)")

        # 1. Derived tensors from patterns (fp32), cached by parameter version.
        A, B_, At, Bt, diag = self._derived_from_patterns(device=device, D=int(D), H=int(H))

        # 2. Project Tag to Real/Imag parts (x, y) and flatten BT for faster bmm.
        bt = int(B) * int(T)
        x = torch.cos(tag_f).unsqueeze(2).expand(int(B), int(T), int(H), int(D)).reshape(bt, int(H), int(D)).contiguous()
        y = torch.sin(tag_f).unsqueeze(2).expand(int(B), int(T), int(H), int(D)).reshape(bt, int(H), int(D)).contiguous()

        # 4. Iterative Dynamics (Settling) using Real-Valued Components
        scale = coupling / float(self.buckets)

        # For Telemetry: track energy and convergence
        # Avoid Python overhead unless aux is requested.
        energy_history: list[float] = []

        # Helper views for batched matmul: (H,D,K) etc.
        At_h = At  # (H,D,K)
        Bt_h = Bt
        A_h = A  # (H,K,D)
        B_h = B_

        for s_idx in range(int(steps)):
            x_prev: Tensor | None = None
            y_prev: Tensor | None = None
            if collect_aux and (s_idx % 5) == 0:
                x_prev = x
                y_prev = y

            # Compute u = P z (K-dim), where P = A + iB and z = x + iy:
            # u_r = x @ A^T - y @ B^T
            # u_i = y @ A^T + x @ B^T
            # Use true batched matmul (bmm) to avoid broadcasting overhead.
            # Shapes:
            #   x_t,y_t: (H, BT, D)
            #   At,Bt:   (H, D, K)
            x_t = x.transpose(0, 1)
            y_t = y.transpose(0, 1)
            u_r_t = torch.bmm(x_t, At_h) - torch.bmm(y_t, Bt_h)  # (H,BT,K)
            u_i_t = torch.bmm(y_t, At_h) + torch.bmm(x_t, Bt_h)  # (H,BT,K)

            # v = P^H u (D-dim):
            # v_r = u_r @ A + u_i @ B
            # v_i = u_i @ A - u_r @ B
            #   u_r_t,u_i_t: (H, BT, K)
            #   A,B:         (H, K, D)
            v_r_t = torch.bmm(u_r_t, A_h) + torch.bmm(u_i_t, B_h)  # (H,BT,D)
            v_i_t = torch.bmm(u_i_t, A_h) - torch.bmm(u_r_t, B_h)  # (H,BT,D)
            v_r = v_r_t.transpose(0, 1).contiguous()  # (BT,H,D)
            v_i = v_i_t.transpose(0, 1).contiguous()  # (BT,H,D)

            if device.type == "cuda":
                if self._update_cuda is None:
                    raise RuntimeError(
                        "CUDA resonant update requested but CUDA/Triton update implementation is unavailable.\n"
                        "Fix: ensure CUDA is available and Triton is installed."
                    )
                x, y = self._update_cuda.forward(
                    x=x,
                    y=y,
                    vr=v_r,
                    vi=v_i,
                    diag=diag,
                    scale=float(scale),
                    damping=float(damping),
                    zero_diag=bool(self.zero_diag),
                )
            elif device.type == "mps":
                if self._update_mps is None:
                    raise RuntimeError(
                        "MPS resonant update requested but Metal update implementation is unavailable.\n"
                        "Fix: ensure MPS is available and the Metal extension can be built."
                    )
                x, y = self._update_mps.forward(
                    x=x,
                    y=y,
                    vr=v_r,
                    vi=v_i,
                    diag=diag,
                    scale=float(scale),
                    damping=float(damping),
                    zero_diag=bool(self.zero_diag),
                )
            else:
                # CPU path: use torch pointwise ops.
                c_r = v_r / float(D)
                c_i = v_i / float(D)
                if self.zero_diag:
                    c_r = c_r - diag.unsqueeze(0) * x
                    c_i = c_i - diag.unsqueeze(0) * y
                x = x * (1.0 - damping) + float(scale) * c_r
                y = y * (1.0 - damping) + float(scale) * c_i
                mag = torch.sqrt(x * x + y * y + 1e-12)
                x = x / mag
                y = y / mag

            if x_prev is not None and y_prev is not None:
                # Track per-iteration change magnitude (more informative than ~1.0 normalization energy).
                delta = ((x - x_prev).pow(2) + (y - y_prev).pow(2)).mean()
                energy_history.append(float(delta.item()))

        # 5. Final Similarity calculation using real components
        xr = x.unsqueeze(-2)  # (BT,H,1,D)
        yi = y.unsqueeze(-2)

        dot_r = (torch.matmul(xr, At) + torch.matmul(yi, Bt)).squeeze(-2)  # (BT,H,K)
        dot_i = (torch.matmul(yi, At) - torch.matmul(xr, Bt)).squeeze(-2)  # (BT,H,K)

        sim = torch.sqrt(dot_r * dot_r + dot_i * dot_i + 1e-12) / float(D)
        sim = sim.clamp(0.0, 1.0)

        logits = sim.view(int(B), int(T), int(H), int(self.buckets))
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

