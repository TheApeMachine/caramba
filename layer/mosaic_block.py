"""MOSAIC block layer: no attention, no KV cache.

Implements a streaming, shape-preserving block that combines:
- Local mixer: depthwise causal conv + gated MLP
- Multiscale continuous state bank: leaky integrators across K timescales
- Hard-addressed associative cache: fixed-size hash table with O(1) read/write

This is an explicit-memory alternative to transformer attention/KV caches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from caramba.carmath import last_write_wins, leaky_integrator_scan
from caramba.config.layer import MosaicBlockLayerConfig
from caramba.console import logger

def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _rms_norm(x: Tensor, *, eps: float = 1e-6) -> Tensor:
    # x: (..., d)
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + float(eps))


@dataclass
class _State:
    # conv buffer holds the last (k-1) normalized vectors: (B, k-1, d_model)
    conv_buf: Tensor
    # multiscale state bank: (B, K, d_model)
    s: Tensor
    # non-decaying register file (optional): (B, R, d_model)
    regs: Tensor | None
    # global step counter (for LRU replacement via per-slot timestamps)
    step: int
    # set-associative hash memory:
    # - keys: (B, H, buckets, assoc, key_dim)
    # - vals: (B, H, buckets, assoc, mem_dim)
    # - last: (B, H, buckets, assoc) int64 timestamps, -1 means empty
    mem_k: Tensor
    mem_v: Tensor
    mem_last: Tensor


def _get_state(ctx: object | None, key: str) -> _State | None:
    if ctx is None:
        return None
    store = getattr(ctx, "_mosaic", None)
    if store is None:
        return None
    if not isinstance(store, dict):
        return None
    st = store.get(key)
    return st if isinstance(st, _State) else None


def _set_state(ctx: object | None, key: str, st: _State) -> None:
    if ctx is None:
        return
    store = getattr(ctx, "_mosaic", None)
    if store is None:
        store = {}
        try:
            setattr(ctx, "_mosaic", store)
        except Exception:
            return
    if isinstance(store, dict):
        store[key] = st


class MosaicBlockLayer(nn.Module):
    """Streaming MOSAIC block (shape preserving)."""

    def __init__(self, config: MosaicBlockLayerConfig) -> None:
        super().__init__()
        self.config = config

        d = int(config.d_model)
        k = int(config.conv_kernel)
        if k < 1:
            raise ValueError(f"conv_kernel must be >= 1, got {k}")

        # Local mixer.
        self.conv = nn.Conv1d(
            in_channels=d,
            out_channels=d,
            kernel_size=k,
            padding=k - 1,  # causal: we'll slice to length T
            groups=d,
            bias=False,
        )
        self.gate_proj = nn.Linear(d, d, bias=True)

        hidden = max(1, int(round(float(config.mlp_mult) * d)))
        self.mlp_up = nn.Linear(d, hidden, bias=True)
        self.mlp_down = nn.Linear(hidden, d, bias=True)
        self.dropout = nn.Dropout(float(config.dropout_p))

        # Multiscale state bank.
        K = int(config.state_k)
        if K < 1:
            raise ValueError(f"state_k must be >= 1, got {K}")
        self.state_k = K
        self.state_in = nn.Linear(d, K * d, bias=False)
        self.state_out = nn.Linear(K * d, d, bias=False)

        # Learnable decays per timescale (scalar per k), constrained to (0,1).
        # Initialize roughly log-spaced between [min,max].
        dmin = float(config.state_decay_min)
        dmax = float(config.state_decay_max)
        if not (0.0 <= dmin <= 1.0 and 0.0 <= dmax <= 1.0 and dmin <= dmax):
            raise ValueError(f"Invalid state_decay range: min={dmin}, max={dmax}")
        if dmin == 0.0 and dmax == 0.0:
            init = torch.zeros(K)
        else:
            # log-space in (0,1): use logit parameterization.
            # Avoid exactly 0/1 to keep logits finite.
            lo = max(1e-4, min(1.0 - 1e-4, dmin))
            hi = max(1e-4, min(1.0 - 1e-4, dmax))
            # If lo==hi, constant decay.
            if abs(lo - hi) < 1e-12:
                decays = torch.full((K,), float(lo))
            else:
                # Geometric spacing in (lo,hi).
                decays = torch.exp(torch.linspace(math.log(lo), math.log(hi), K))
            init = torch.log(decays) - torch.log1p(-decays)  # logit
        self.state_decay_logit = nn.Parameter(init)

        # Memory routing + storage.
        buckets = int(config.mem_buckets)
        if buckets < 2:
            raise ValueError(f"mem_buckets must be >= 2, got {buckets}")
        hashes = int(config.mem_hashes)
        if hashes < 1:
            raise ValueError(f"mem_hashes must be >= 1, got {hashes}")
        mem_dim = int(config.mem_dim)
        if mem_dim < 1:
            raise ValueError(f"mem_dim must be >= 1, got {mem_dim}")
        self.mem_buckets = buckets
        self.mem_hashes = hashes
        self.mem_dim = mem_dim
        self.mem_router = str(getattr(config, "mem_router", "bits")).lower().strip()
        if self.mem_router not in ("bits", "vq"):
            raise ValueError(f"Unsupported mem_router={self.mem_router!r}")

        # Router: "bits" (learned sign bits) or "vq" (product-quantized VQ routing).
        self.mem_bits = int((buckets - 1).bit_length())
        self.mem_read_bits: nn.Linear | None = None
        self.mem_write_bits: nn.Linear | None = None

        # VQ router params (created only when enabled).
        self.mem_vq_groups = int(getattr(config, "mem_vq_groups", 2))
        self.mem_vq_codebook_size = int(getattr(config, "mem_vq_codebook_size", 256))
        self.mem_vq_group_dim = int(getattr(config, "mem_vq_group_dim", 16))
        self.mem_vq_beam = int(getattr(config, "mem_vq_beam", 1))
        self.mem_write_multi = bool(getattr(config, "mem_write_multi", False))
        self.mem_vq_proj_r: nn.Linear | None = None
        self.mem_vq_proj_w: nn.Linear | None = None
        self.mem_vq_codebook_r: nn.Parameter | None = None
        self.mem_vq_codebook_w: nn.Parameter | None = None
        self._vq_powK_cache: Tensor | None = None
        self._vq_combo_idx: Tensor | None = None

        if self.mem_router == "bits":
            # Bit-projection matrices for read/write keys.
            self.mem_read_bits = nn.Linear(d, hashes * self.mem_bits, bias=False)
            self.mem_write_bits = nn.Linear(d, hashes * self.mem_bits, bias=False)
        else:
            # VQ routing uses a low-dimensional projection and G independent codebooks.
            G = int(self.mem_vq_groups)
            Kc = int(self.mem_vq_codebook_size)
            gd = int(self.mem_vq_group_dim)
            if G < 1 or Kc < 2 or gd < 1:
                raise ValueError("Invalid VQ router config")
            route_dim = int(G * gd)
            self.mem_vq_proj_r = nn.Linear(d, hashes * route_dim, bias=False)
            self.mem_vq_proj_w = nn.Linear(d, hashes * route_dim, bias=False)
            # Codebooks: (hashes, G, K, gd)
            self.mem_vq_codebook_r = nn.Parameter(torch.randn(hashes, G, Kc, gd) * 0.02)
            self.mem_vq_codebook_w = nn.Parameter(torch.randn(hashes, G, Kc, gd) * 0.02)

            # Precompute powK for base-K addressing (on device lazily).
            self.mem_bits = int((buckets - 1).bit_length())  # keep for teacher bit loss compatibility
        # Within-bucket fuzzy routing keys (set-associative).
        assoc = int(getattr(config, "mem_assoc", 1))
        if assoc < 1:
            raise ValueError(f"mem_assoc must be >= 1, got {assoc}")
        self.mem_assoc = assoc
        key_dim = int(getattr(config, "mem_key_dim", 0) or 0)
        if key_dim < 1:
            raise ValueError(f"mem_key_dim must be >= 1, got {key_dim}")
        self.mem_key_dim = key_dim
        self.mem_qkey = nn.Linear(d, key_dim, bias=False)
        self.mem_wkey = nn.Linear(d, key_dim, bias=False)
        self.mem_value = nn.Linear(d, mem_dim, bias=False)
        self.mem_out = nn.Linear(mem_dim, d, bias=False)
        self.mem_write_gate = nn.Linear(d, 1, bias=True)
        # Predict "will this write be useful soon?" (curriculum aux).
        self.mem_utility_head = nn.Linear(d, 1, bias=True)

        # Optional dVM register file (non-decaying scratchpad).
        reg_slots = getattr(config, "reg_slots", None)
        self.reg_slots = int(reg_slots) if reg_slots is not None else 0
        self.reg_write_gate: nn.Linear | None = None
        self.reg_sel: nn.Linear | None = None
        self.reg_value: nn.Linear | None = None
        self.gate_reg: nn.Linear | None = None
        if self.reg_slots > 0:
            self.reg_write_gate = nn.Linear(d, 1, bias=True)
            self.reg_sel = nn.Linear(d, self.reg_slots, bias=True)
            self.reg_value = nn.Linear(d, d, bias=True)
            self.gate_reg = nn.Linear(d, 1, bias=True)
            with torch.no_grad():
                self.gate_reg.bias.fill_(float(getattr(config, "gate_reg_init", 0.0)))

        # Optional opcode head (logging / supervision). Convention:
        # 0=Nop, 1=Read, 2=Write, 3=Clear.
        self.opcodes_enabled = bool(getattr(config, "opcodes_enabled", False))
        self.opcode_vocab = int(getattr(config, "opcode_vocab", 4))
        self.opcode_head: nn.Linear | None = None
        if self.opcodes_enabled:
            if self.opcode_vocab < 2:
                raise ValueError(f"opcode_vocab must be >= 2, got {self.opcode_vocab}")
            self.opcode_head = nn.Linear(d, self.opcode_vocab, bias=True)

        # Fusion gates.
        self.gate_long = nn.Linear(d, 1, bias=True)
        self.gate_mem = nn.Linear(d, 1, bias=True)
        with torch.no_grad():
            self.gate_long.bias.fill_(float(config.gate_long_init))
            self.gate_mem.bias.fill_(float(config.gate_mem_init))

        # Stable per-module key for ctx storage.
        self._ctx_key = f"mosaic_block::{id(self)}"

        # Cached powers for bit-packing (constructed lazily on device/dtype).
        self._bit_pows: Tensor | None = None

    def _vq_powK_tensor(self, device: torch.device) -> Tensor:
        """Return (G,) powers of K for base-K addressing."""
        p = self._vq_powK_cache
        if p is None or p.device != device or p.numel() != int(self.mem_vq_groups):
            Kc = int(self.mem_vq_codebook_size)
            G = int(self.mem_vq_groups)
            p = torch.tensor([Kc**i for i in range(G)], device=device, dtype=torch.long)
            self._vq_powK_cache = p
        return p

    def _vq_combo_index(self, device: torch.device) -> Tensor:
        """Return (Ncand, G) indices selecting beam options per group."""
        beam = int(self.mem_vq_beam)
        G = int(self.mem_vq_groups)
        if beam <= 1:
            idx = torch.zeros((1, G), device=device, dtype=torch.long)
            return idx
        cached = self._vq_combo_idx
        if cached is not None and cached.device == device and cached.shape[1] == G:
            return cached
        # Cartesian product of range(beam) repeated G times.
        grids = torch.cartesian_prod(*([torch.arange(beam, device=device)] * G))
        self._vq_combo_idx = grids.to(dtype=torch.long)
        return self._vq_combo_idx

    def _route_bits(self, z: Tensor) -> Tensor:
        """Sign-bit routing: z is (B,H,BITS) -> idx (B,H)."""
        bits = z > 0
        idx = (bits.to(torch.long) * self._bit_powers(z.device)).sum(dim=-1)  # (B,H)
        if _is_power_of_two(self.mem_buckets) and (1 << self.mem_bits) == self.mem_buckets:
            return idx
        return torch.remainder(idx, int(self.mem_buckets))

    def _route_vq(
        self,
        y: Tensor,
        *,
        codebook: Tensor,
        return_group_logits: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        """Product-quantized VQ routing.

        Args:
          y: (B,H,G,gd) projected routing vectors
          codebook: (H,G,K,gd)

        Returns:
          idx: (B,H) primary bucket indices
          cand_idx: (B,H,Ncand) candidate indices when beam>1 else None
          cand_w: (B,H,Ncand) weights for candidates (product of per-group softmax probs) else None
        """
        B, Hh, G, gd = y.shape
        Kc = int(codebook.size(2))
        beam = int(self.mem_vq_beam)
        # Compute squared distances: (B,H,G,K)
        # dist = ||y||^2 + ||e||^2 - 2 y·e
        y2 = (y * y).sum(dim=-1, keepdim=True)  # (B,H,G,1)
        e = codebook  # (H,G,K,gd)
        e2 = (e * e).sum(dim=-1).view(1, Hh, G, Kc)  # (1,H,G,K)
        dot = torch.einsum("bhgd,hgkd->bhgk", y, e)  # (B,H,G,K)
        dist = y2.view(B, Hh, G, 1) + e2 - 2.0 * dot
        group_logits: Tensor | None = None
        if bool(return_group_logits):
            # Use negative squared distance as logits for CE supervision.
            group_logits = (-dist).to(dtype=torch.float32)

        # Top-1 codes per group (nearest).
        best = dist.argmin(dim=-1)  # (B,H,G)
        powK = self._vq_powK_tensor(y.device).view(1, 1, G)  # (1,1,G)
        idx = (best.to(torch.long) * powK).sum(dim=-1)  # (B,H)
        if not (_is_power_of_two(self.mem_buckets) and int(self.mem_buckets) == int(self.mem_vq_codebook_size) ** int(self.mem_vq_groups)):
            idx = torch.remainder(idx, int(self.mem_buckets))

        if beam <= 1:
            return idx, None, None, group_logits

        # Candidate buckets: top-beam codes per group.
        topk = min(int(beam), Kc)
        top_dist, top_idx = torch.topk(dist, k=topk, dim=-1, largest=False)  # (B,H,G,beam)
        # Convert distances to per-group probabilities (softmax over beam).
        p = torch.softmax((-top_dist).float(), dim=-1).to(dtype=y.dtype)  # (B,H,G,beam)

        combos = self._vq_combo_index(y.device)  # (Ncand,G) selecting [0..beam)
        Nc = int(combos.size(0))
        # Gather codes/probs per combo.
        # codes_sel: (B,H,Nc,G)
        codes_sel = []
        probs_sel = []
        for g in range(G):
            sel = combos[:, g].view(1, 1, Nc).expand(B, Hh, Nc)  # (B,H,Nc)
            cg = top_idx[:, :, g, :]  # (B,H,beam)
            pg = p[:, :, g, :]        # (B,H,beam)
            codes_sel.append(torch.gather(cg, dim=-1, index=sel))
            probs_sel.append(torch.gather(pg, dim=-1, index=sel))
        codes = torch.stack(codes_sel, dim=-1)  # (B,H,Nc,G)
        probs = torch.stack(probs_sel, dim=-1)  # (B,H,Nc,G)
        cand_w = probs.prod(dim=-1)              # (B,H,Nc)
        cand_idx = (codes.to(torch.long) * powK.view(1, 1, 1, G)).sum(dim=-1)  # (B,H,Nc)
        cand_idx = torch.remainder(cand_idx, int(self.mem_buckets))
        return idx, cand_idx, cand_w, group_logits

    def _bit_powers(self, device: torch.device) -> Tensor:
        p = self._bit_pows
        if p is None or p.device != device or p.numel() != self.mem_bits:
            p = (2 ** torch.arange(self.mem_bits, device=device, dtype=torch.long)).view(1, 1, self.mem_bits)
            self._bit_pows = p
        return p

    def _local_mixer(self, u: Tensor, *, ctx_state: _State | None) -> tuple[Tensor, Tensor | None]:
        """Local mixer output + updated conv buffer (if streaming)."""
        B, T, D = u.shape
        k = int(self.config.conv_kernel)

        # Two modes:
        # - Full-sequence (training/prefill): vectorized conv with causal padding.
        # - Streaming decode (T==1 with ctx): conv over the last k tokens (buffer + current).
        new_buf: Tensor | None = None
        if ctx_state is not None and int(T) == 1 and k > 1:
            # window: (B, k, D) = (B, k-1, D) + (B, 1, D)
            window = torch.cat([ctx_state.conv_buf.to(dtype=u.dtype, device=u.device), u], dim=1)
            # Depthwise conv1d with no padding: output length is 1.
            x = F.conv1d(
                window.transpose(1, 2),
                self.conv.weight,
                bias=None,
                stride=1,
                padding=0,
                dilation=1,
                groups=D,
            ).transpose(1, 2)
            # Update buffer: keep last (k-1) normalized vectors.
            new_buf = window[:, 1:, :].detach()
        else:
            # Full sequence conv (vectorized).
            x = self.conv(u.transpose(1, 2))[:, :, :T].transpose(1, 2)  # causal slice
            # Buffer update: keep last (k-1) normalized vectors.
            if ctx_state is not None and k > 1:
                keep = min(k - 1, int(T))
                new_buf = u[:, -keep:, :].detach() if keep > 0 else ctx_state.conv_buf

        gate = torch.sigmoid(self.gate_proj(x))
        x = x * gate
        x = self.mlp_down(torch.nn.functional.silu(self.mlp_up(x)))
        x = self.dropout(x)
        return x, new_buf

    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        # x: (B, T, d_model)
        B, T, D = x.shape
        if D != int(self.config.d_model):
            raise ValueError(f"Expected d_model={int(self.config.d_model)}, got {D}")

        # Fetch or init persistent state (for streaming decode / multi-call inference).
        st = _get_state(ctx, self._ctx_key)
        if st is not None:
            # Re-init if batch size changes.
            bad_regs = bool(isinstance(st.regs, Tensor) and st.regs.size(0) != B)
            if st.conv_buf.size(0) != B or st.s.size(0) != B or st.mem_k.size(0) != B or bad_regs:
                st = None

        if st is None:
            k = int(self.config.conv_kernel)
            conv_buf = x.new_zeros((B, max(0, k - 1), D))
            s = x.new_zeros((B, self.state_k, D))
            regs = x.new_zeros((B, self.reg_slots, D)) if self.reg_slots > 0 else None
            step = 0
            mem_k = x.new_zeros((B, self.mem_hashes, self.mem_buckets, self.mem_assoc, self.mem_key_dim))
            mem_v = x.new_zeros((B, self.mem_hashes, self.mem_buckets, self.mem_assoc, self.mem_dim))
            mem_last = torch.full(
                (B, self.mem_hashes, self.mem_buckets, self.mem_assoc),
                -1,
                device=x.device,
                dtype=torch.long,
            )
            st = _State(conv_buf=conv_buf, s=s, regs=regs, step=step, mem_k=mem_k, mem_v=mem_v, mem_last=mem_last)

        # Pre-norm.
        u = _rms_norm(x)

        # Local mixer (vectorized), plus keep conv buffer for streaming compatibility.
        local, new_buf = self._local_mixer(u, ctx_state=st)

        # Optional teacher controls (Stage D1/D2 curriculum).
        teacher: dict[str, Tensor] | None = None
        if ctx is not None:
            t = getattr(ctx, "mosaic_teacher", None)
            if isinstance(t, dict) and len(t) > 0:
                teacher = t
        teacher_p = 1.0
        if ctx is not None:
            try:
                teacher_p = float(getattr(ctx, "mosaic_teacher_p", 1.0))
            except Exception:
                teacher_p = 1.0
        teacher_p = max(0.0, min(1.0, teacher_p))

        # Forced-read dropout: drop local mixer contribution to force memory/state use.
        drop_p = float(getattr(self.config, "forced_read_dropout_p", 0.0))
        drop_mask: Tensor | None = None
        if ctx is not None:
            dm = getattr(ctx, "mosaic_drop_local", None)
            if isinstance(dm, Tensor):
                # Accept (B,T) or (T,) masks.
                try:
                    if dm.dim() == 2 and dm.size(0) == B and dm.size(1) == T:
                        drop_mask = dm.to(device=x.device, dtype=x.dtype).view(B, T, 1)
                    elif dm.dim() == 1 and dm.size(0) == T:
                        drop_mask = dm.to(device=x.device, dtype=x.dtype).view(1, T, 1).expand(B, T, 1)
                except Exception:
                    drop_mask = None
        if drop_mask is None and self.training and drop_p > 0.0:
            # Tokenwise Bernoulli drop (simple v1). Spans can be layered later.
            drop_mask = (torch.rand((B, T, 1), device=x.device) < drop_p).to(dtype=x.dtype)
        if drop_mask is not None:
            local = local * (1.0 - drop_mask)

        # Multiscale state + hash memory are applied in a causal scan to match streaming semantics.
        decay = torch.sigmoid(self.state_decay_logit).to(dtype=x.dtype, device=x.device).view(1, self.state_k, 1)
        eta = float(self.config.mem_write_eta)
        thr = float(self.config.mem_write_threshold)

        g_seq = x.new_empty((B, T, D))
        r_seq = x.new_empty((B, T, D))
        z_seq = x.new_zeros((B, T, D)) if self.reg_slots > 0 else None

        s = st.s
        regs = st.regs
        # Persistent hash memory is *not* part of the differentiable computation graph.
        # Treat it as a mutable cache/state. Keeping it detached avoids autograd
        # versioning errors when we update it in-place (especially on MPS).
        mem_k = st.mem_k.detach()
        mem_v = st.mem_v.detach()
        mem_last = st.mem_last.detach()

        batch_idx = torch.arange(B, device=x.device, dtype=torch.long)
        hash_idx = torch.arange(self.mem_hashes, device=x.device, dtype=torch.long).view(1, -1).expand(B, -1)

        temp = float(getattr(self.config, "mem_read_temp", 1.0))
        temp = max(1e-6, temp)
        match_thr = float(getattr(self.config, "mem_match_threshold", 0.0))
        key_scale = 1.0 / math.sqrt(float(self.mem_key_dim))

        # Optional aux outputs for curriculum losses (gate/bits imitation).
        collect_aux = bool(getattr(ctx, "mosaic_collect_aux", False)) if ctx is not None else False
        if collect_aux and ctx is not None:
            try:
                setattr(ctx, "mosaic_aux_out", {})  # overwritten at end
            except Exception:
                collect_aux = False
        if collect_aux:
            gate_logits_seq = x.new_empty((B, T))
            util_logits_seq = x.new_empty((B, T))
            opcode_logits_seq = x.new_empty((B, T, self.opcode_vocab), dtype=torch.float32) if self.opcodes_enabled else None
            reg_write_gate_logits_seq = x.new_empty((B, T)) if self.reg_slots > 0 else None
            reg_sel_logits_seq = x.new_empty((B, T, self.reg_slots), dtype=torch.float32) if self.reg_slots > 0 else None
            # Router aux depends on router kind.
            if self.mem_router == "bits":
                read_bit_logits_seq = x.new_empty((B, T, self.mem_hashes, self.mem_bits))
                write_bit_logits_seq = x.new_empty((B, T, self.mem_hashes, self.mem_bits))
                read_codes_seq = None
                write_codes_seq = None
                read_vq_logits_seq = None
                write_vq_logits_seq = None
            else:
                read_bit_logits_seq = None
                write_bit_logits_seq = None
                read_codes_seq = x.new_empty((B, T, self.mem_hashes, int(self.mem_vq_groups)), dtype=torch.long)
                write_codes_seq = x.new_empty((B, T, self.mem_hashes, int(self.mem_vq_groups)), dtype=torch.long)
                read_vq_logits_seq = x.new_empty(
                    (B, T, self.mem_hashes, int(self.mem_vq_groups), int(self.mem_vq_codebook_size)),
                    dtype=torch.float32,
                )
                write_vq_logits_seq = x.new_empty(
                    (B, T, self.mem_hashes, int(self.mem_vq_groups), int(self.mem_vq_codebook_size)),
                    dtype=torch.float32,
                )

        def _finalize() -> Tensor:
            # Contrastive auxiliary: make memory reads predictive of future hidden state.
            contrastive_loss: Tensor | None = None
            if collect_aux:
                try:
                    delta = int(getattr(self.config, "aux_contrastive_delta", 1))
                except Exception:
                    delta = 1
                if delta > 0 and delta < int(T):
                    # Prefer supervising only where a teacher read was requested.
                    mask_pos: Tensor | None = None
                    if teacher is not None and "read_bucket" in teacher and isinstance(teacher["read_bucket"], Tensor):
                        tb = teacher["read_bucket"]
                        try:
                            if tb.dim() == 2 and tb.size(0) == B and tb.size(1) == T:
                                mask_pos = tb >= 0
                            elif tb.dim() == 3 and tb.size(0) == B and tb.size(1) == T:
                                mask_pos = (tb >= 0).any(dim=-1)
                        except Exception:
                            mask_pos = None
                    if mask_pos is None:
                        mask_pos = torch.ones((B, T), device=x.device, dtype=torch.bool)

                    # Only positions where t+delta exists.
                    mask_pos = mask_pos[:, : T - delta]
                    idx = torch.nonzero(mask_pos, as_tuple=False)
                    if idx.numel() > 0:
                        # Subsample to keep cost bounded.
                        max_n = 256
                        if idx.size(0) > max_n:
                            perm = torch.randperm(idx.size(0), device=idx.device)[:max_n]
                            idx = idx[perm]
                        b_idx = idx[:, 0]
                        t_idx = idx[:, 1]
                        r_sel = r_seq[b_idx, t_idx, :].float()
                        p_sel = u[b_idx, t_idx + delta, :].float()
                        # InfoNCE within this minibatch: positives are aligned pairs.
                        logits = (r_sel @ p_sel.t()) * (1.0 / math.sqrt(float(r_sel.size(-1))))
                        targets = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
                        contrastive_loss = F.cross_entropy(logits, targets)

            # Fusion gates (token-wise).
            gate_long = torch.sigmoid(self.gate_long(u))  # (B,T,1)
            gate_mem = torch.sigmoid(self.gate_mem(u))    # (B,T,1)
            gate_reg = torch.sigmoid(self.gate_reg(u)) if (self.gate_reg is not None and z_seq is not None) else None

            y = x + local + gate_long * g_seq + gate_mem * r_seq
            if gate_reg is not None and z_seq is not None:
                y = y + gate_reg * z_seq

            # Persist updated state.
            st.s = s.detach()
            if self.reg_slots > 0 and isinstance(regs, Tensor):
                st.regs = regs.detach()
            st.mem_k = mem_k.detach()
            st.mem_v = mem_v.detach()
            st.mem_last = mem_last.detach()
            if new_buf is not None:
                st.conv_buf = new_buf
            # Emit aux outputs (best-effort) for curriculum objectives.
            if collect_aux and ctx is not None:
                try:
                    setattr(
                        ctx,
                        "mosaic_aux_out",
                        {
                            "mosaic_write_gate_logits": gate_logits_seq.detach(),
                            "mosaic_write_utility_logits": util_logits_seq.detach(),
                            **(
                                {"mosaic_opcode_logits": opcode_logits_seq.detach()}
                                if isinstance(opcode_logits_seq, Tensor)
                                else {}
                            ),
                            **(
                                {"mosaic_reg_write_gate_logits": reg_write_gate_logits_seq.detach()}
                                if isinstance(reg_write_gate_logits_seq, Tensor)
                                else {}
                            ),
                            **(
                                {"mosaic_reg_sel_logits": reg_sel_logits_seq.detach()}
                                if isinstance(reg_sel_logits_seq, Tensor)
                                else {}
                            ),
                            **(
                                {"mosaic_regs_last": regs.detach()}
                                if isinstance(regs, Tensor)
                                else {}
                            ),
                            **(
                                {"mosaic_contrastive_loss": contrastive_loss.detach()}
                                if isinstance(contrastive_loss, Tensor)
                                else {}
                            ),
                            **(
                                {
                                    "mosaic_read_bit_logits": read_bit_logits_seq.detach(),
                                    "mosaic_write_bit_logits": write_bit_logits_seq.detach(),
                                }
                                if isinstance(read_bit_logits_seq, Tensor) and isinstance(write_bit_logits_seq, Tensor)
                                else {}
                            ),
                            **(
                                {
                                    "mosaic_vq_read_logits": read_vq_logits_seq.detach(),
                                    "mosaic_vq_write_logits": write_vq_logits_seq.detach(),
                                }
                                if isinstance(read_vq_logits_seq, Tensor) and isinstance(write_vq_logits_seq, Tensor)
                                else {}
                            ),
                            **(
                                {
                                    "mosaic_read_codes": read_codes_seq.detach(),
                                    "mosaic_write_codes": write_codes_seq.detach(),
                                }
                                if isinstance(read_codes_seq, Tensor) and isinstance(write_codes_seq, Tensor)
                                else {}
                            ),
                        },
                    )
                except Exception:
                    pass
            _set_state(ctx, self._ctx_key, st)
            return y

        # ---------------------------------------------------------------------
        # Fast training path (chunked; avoids Python per-token loop)
        # ---------------------------------------------------------------------
        stats_enabled = bool(ctx is not None and bool(getattr(ctx, "mosaic_stats_enabled", False)))
        has_clear = bool(isinstance(teacher, dict) and ("clear" in teacher))
        # Registers currently require strict sequential semantics; disable the chunked path.
        use_fast_train = (
            bool(self.training)
            and int(T) > 1
            and (not stats_enabled)
            and (not has_clear)
            and (self.reg_slots <= 0)
        )
        if use_fast_train:
            Hh = int(self.mem_hashes)
            A = int(self.mem_assoc)
            key_dim = int(self.mem_key_dim)
            mem_dim = int(self.mem_dim)

            # Best-effort debug marker.
            if ctx is not None:
                try:
                    setattr(ctx, "mosaic_fast_train_used", True)
                    setattr(ctx, "mosaic_fast_train_steps", int(getattr(ctx, "mosaic_fast_train_steps", 0)) + 1)
                except Exception:
                    pass

            # Router results for the whole sequence.
            idx_r_all: Tensor
            idx_w_all: Tensor
            cand_r_idx_all: Tensor | None = None
            cand_r_w_all: Tensor | None = None
            cand_w_idx_all: Tensor | None = None
            cand_w_w_all: Tensor | None = None

            if self.mem_router == "bits":
                assert self.mem_read_bits is not None and self.mem_write_bits is not None
                z_read_all = self.mem_read_bits(u).view(B, T, Hh, int(self.mem_bits))
                z_write_all = self.mem_write_bits(u).view(B, T, Hh, int(self.mem_bits))
                if collect_aux and read_bit_logits_seq is not None and write_bit_logits_seq is not None:
                    read_bit_logits_seq[:, :, :, :] = z_read_all
                    write_bit_logits_seq[:, :, :, :] = z_write_all
                idx_r_all = self._route_bits(z_read_all.reshape(B * T, Hh, int(self.mem_bits))).view(B, T, Hh)
                idx_w_all = self._route_bits(z_write_all.reshape(B * T, Hh, int(self.mem_bits))).view(B, T, Hh)
            else:
                # VQ router (vectorized across the whole sequence).
                assert self.mem_vq_proj_r is not None and self.mem_vq_proj_w is not None
                assert self.mem_vq_codebook_r is not None and self.mem_vq_codebook_w is not None
                G = int(self.mem_vq_groups)
                gd = int(self.mem_vq_group_dim)
                route_dim = int(G * gd)

                yr_all = self.mem_vq_proj_r(u).view(B, T, Hh, route_dim).view(B * T, Hh, G, gd)
                yw_all = self.mem_vq_proj_w(u).view(B, T, Hh, route_dim).view(B * T, Hh, G, gd)

                idx_r_flat, cand_r_idx, cand_r_w, gl_r = self._route_vq(
                    yr_all,
                    codebook=self.mem_vq_codebook_r,
                    return_group_logits=bool(collect_aux),
                )
                idx_w_flat, cand_w_idx, cand_w_w, gl_w = self._route_vq(
                    yw_all,
                    codebook=self.mem_vq_codebook_w,
                    return_group_logits=bool(collect_aux),
                )
                idx_r_all = idx_r_flat.view(B, T, Hh)
                idx_w_all = idx_w_flat.view(B, T, Hh)
                if isinstance(cand_r_idx, Tensor) and isinstance(cand_r_w, Tensor):
                    cand_r_idx_all = cand_r_idx.view(B, T, Hh, -1)
                    cand_r_w_all = cand_r_w.view(B, T, Hh, -1)
                if isinstance(cand_w_idx, Tensor) and isinstance(cand_w_w, Tensor):
                    cand_w_idx_all = cand_w_idx.view(B, T, Hh, -1)
                    cand_w_w_all = cand_w_w.view(B, T, Hh, -1)

                if collect_aux and isinstance(read_vq_logits_seq, Tensor) and isinstance(write_vq_logits_seq, Tensor):
                    if isinstance(gl_r, Tensor):
                        read_vq_logits_seq[:, :, :, :, :] = gl_r.view(B, T, Hh, G, int(self.mem_vq_codebook_size))
                    if isinstance(gl_w, Tensor):
                        write_vq_logits_seq[:, :, :, :, :] = gl_w.view(B, T, Hh, G, int(self.mem_vq_codebook_size))

                if collect_aux and isinstance(read_codes_seq, Tensor) and isinstance(write_codes_seq, Tensor):
                    # Store top-1 VQ codes per group for introspection.
                    try:
                        e_r = self.mem_vq_codebook_r  # (H,G,K,gd)
                        y2 = (yr_all * yr_all).sum(dim=-1, keepdim=True)  # (BT,H,G,1)
                        e2 = (e_r * e_r).sum(dim=-1).view(1, Hh, G, int(self.mem_vq_codebook_size))
                        dot = torch.einsum("bhgd,hgkd->bhgk", yr_all, e_r)
                        dist = y2 + e2 - 2.0 * dot
                        read_codes_seq[:, :, :, :] = dist.argmin(dim=-1).view(B, T, Hh, G)

                        e_w = self.mem_vq_codebook_w
                        y2w = (yw_all * yw_all).sum(dim=-1, keepdim=True)
                        e2w = (e_w * e_w).sum(dim=-1).view(1, Hh, G, int(self.mem_vq_codebook_size))
                        dotw = torch.einsum("bhgd,hgkd->bhgk", yw_all, e_w)
                        distw = y2w + e2w - 2.0 * dotw
                        write_codes_seq[:, :, :, :] = distw.argmin(dim=-1).view(B, T, Hh, G)
                    except Exception:
                        pass

            # Chunk size (training-only).
            chunk_size = int(getattr(self.config, "mem_train_chunk_size", 128))
            chunk_size = max(1, chunk_size)

            step0 = int(st.step)
            big = int(step0 + int(T) + 1)

            def _teacher_use_mask(*, B: int, C: int) -> Tensor | None:
                if teacher is None:
                    return None
                if teacher_p <= 0.0:
                    return torch.zeros((B, C), device=x.device, dtype=torch.bool)
                if teacher_p >= 1.0:
                    return torch.ones((B, C), device=x.device, dtype=torch.bool)
                return (torch.rand((B, C), device=x.device) < float(teacher_p))

            for t0 in range(0, int(T), int(chunk_size)):
                t1 = min(int(T), t0 + int(chunk_size))
                C = int(t1 - t0)
                u_c = u[:, t0:t1, :]  # (B,C,D)

                # --- State bank (vectorized scan) ---
                inp = self.state_in(u_c).view(B, C, self.state_k, D).permute(0, 2, 1, 3)  # (B,K,C,D)
                s_seq_c, s_last = leaky_integrator_scan(inp, s, decay)
                s = s_last.to(dtype=x.dtype)
                g_seq[:, t0:t1, :] = self.state_out(
                    s_seq_c.permute(0, 2, 1, 3).to(dtype=x.dtype).reshape(B, C, self.state_k * D)
                )

                # Utility head.
                util_logit_c = self.mem_utility_head(u_c).squeeze(-1)
                if collect_aux:
                    util_logits_seq[:, t0:t1] = util_logit_c

                # --- Memory read ---
                idx_r_c = idx_r_all[:, t0:t1, :]  # (B,C,H)
                use_teacher = _teacher_use_mask(B=B, C=C)
                if teacher is not None and "read_bucket" in teacher and isinstance(teacher["read_bucket"], Tensor):
                    tb = teacher["read_bucket"]
                    try:
                        if tb.dim() == 2 and tb.size(0) == B and tb.size(1) == T:
                            tb_c = tb[:, t0:t1].view(B, C, 1).expand(B, C, Hh)
                        elif tb.dim() == 3 and tb.size(0) == B and tb.size(1) == T and tb.size(2) == Hh:
                            tb_c = tb[:, t0:t1, :].view(B, C, Hh)
                        else:
                            tb_c = None
                        if tb_c is not None:
                            m = tb_c >= 0
                            if isinstance(use_teacher, Tensor):
                                m = m & use_teacher.unsqueeze(-1)
                            idx_r_c = torch.where(m, tb_c.to(dtype=idx_r_c.dtype, device=idx_r_c.device), idx_r_c)
                    except Exception:
                        pass

                qk_c = self.mem_qkey(u_c)  # (B,C,key_dim)
                if self.mem_router == "vq" and isinstance(cand_r_idx_all, Tensor) and isinstance(cand_r_w_all, Tensor):
                    cand_idx_c = cand_r_idx_all[:, t0:t1, :, :]  # (B,C,H,Nc)
                    cand_w_c = cand_r_w_all[:, t0:t1, :, :]      # (B,C,H,Nc)
                    ww = cand_w_c / cand_w_c.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                    read_h = u_c.new_zeros((B, Hh, C, mem_dim))
                    Nc = int(cand_idx_c.size(-1))
                    for ci in range(Nc):
                        idx_ci = cand_idx_c[:, :, :, ci]  # (B,C,H)
                        idx_g = idx_ci.permute(0, 2, 1).to(dtype=torch.long).unsqueeze(-1).unsqueeze(-1)  # (B,H,C,1,1)
                        bk = mem_k.gather(dim=2, index=idx_g.expand(B, Hh, C, A, key_dim))
                        bv = mem_v.gather(dim=2, index=idx_g.expand(B, Hh, C, A, mem_dim))
                        bl = mem_last.gather(dim=2, index=idx_g[..., 0].expand(B, Hh, C, A))
                        valid = bl >= 0
                        sim = (bk * qk_c.view(B, 1, C, 1, key_dim)).sum(dim=-1) * float(key_scale)
                        sim = sim.masked_fill(~valid, float("-inf"))
                        any_valid = valid.any(dim=-1, keepdim=True)
                        wslot = torch.softmax(sim / float(temp), dim=-1)
                        wslot = torch.where(any_valid, wslot, torch.zeros_like(wslot))
                        rh_ci = (wslot.unsqueeze(-1) * bv).sum(dim=3)  # (B,H,C,mem_dim)
                        read_h = read_h + rh_ci * ww[:, :, :, ci].permute(0, 2, 1).unsqueeze(-1)
                    r_seq[:, t0:t1, :] = self.mem_out(read_h.sum(dim=1))
                else:
                    idx_g = idx_r_c.permute(0, 2, 1).to(dtype=torch.long).unsqueeze(-1).unsqueeze(-1)  # (B,H,C,1,1)
                    bk = mem_k.gather(dim=2, index=idx_g.expand(B, Hh, C, A, key_dim))
                    bv = mem_v.gather(dim=2, index=idx_g.expand(B, Hh, C, A, mem_dim))
                    bl = mem_last.gather(dim=2, index=idx_g[..., 0].expand(B, Hh, C, A))
                    valid = bl >= 0
                    sim = (bk * qk_c.view(B, 1, C, 1, key_dim)).sum(dim=-1) * float(key_scale)
                    sim = sim.masked_fill(~valid, float("-inf"))
                    any_valid = valid.any(dim=-1, keepdim=True)
                    w = torch.softmax(sim / float(temp), dim=-1)
                    w = torch.where(any_valid, w, torch.zeros_like(w))
                    read_h = (w.unsqueeze(-1) * bv).sum(dim=3)  # (B,H,C,mem_dim)
                    r_seq[:, t0:t1, :] = self.mem_out(read_h.sum(dim=1))

                # --- Memory write (aggregated per slot; last-write-wins) ---
                gate_logit_c = self.mem_write_gate(u_c).squeeze(-1)  # (B,C)
                if collect_aux:
                    gate_logits_seq[:, t0:t1] = gate_logit_c
                p_c = torch.sigmoid(gate_logit_c)  # (B,C)
                mask_c = (p_c > float(thr)).to(dtype=u_c.dtype)
                if teacher is not None and "write_gate" in teacher and isinstance(teacher["write_gate"], Tensor):
                    tg = teacher["write_gate"]
                    try:
                        if tg.dim() == 2 and tg.size(0) == B and tg.size(1) == T:
                            tg_c = tg[:, t0:t1].view(B, C)
                        else:
                            tg_c = None
                        if tg_c is not None:
                            use = tg_c >= 0
                            if isinstance(use_teacher, Tensor):
                                use = use & use_teacher
                            mask_c = torch.where(use, (tg_c > 0).to(dtype=mask_c.dtype, device=mask_c.device), mask_c)
                    except Exception:
                        pass
                w_eta_c = (float(eta) * p_c).to(dtype=u_c.dtype) * mask_c  # (B,C)

                do = w_eta_c > 0
                pos = torch.nonzero(do, as_tuple=False)
                if pos.numel() > 0:
                    b_ev_all = pos[:, 0]
                    t_ev_all = pos[:, 1]

                    idx_w_c = idx_w_all[:, t0:t1, :]  # (B,C,H)
                    if teacher is not None and "write_bucket" in teacher and isinstance(teacher["write_bucket"], Tensor):
                        tbw = teacher["write_bucket"]
                        try:
                            if tbw.dim() == 2 and tbw.size(0) == B and tbw.size(1) == T:
                                tbw_c = tbw[:, t0:t1].view(B, C, 1).expand(B, C, Hh)
                            elif tbw.dim() == 3 and tbw.size(0) == B and tbw.size(1) == T and tbw.size(2) == Hh:
                                tbw_c = tbw[:, t0:t1, :].view(B, C, Hh)
                            else:
                                tbw_c = None
                            if tbw_c is not None:
                                m = tbw_c >= 0
                                if isinstance(use_teacher, Tensor):
                                    m = m & use_teacher.unsqueeze(-1)
                                idx_w_c = torch.where(m, tbw_c.to(dtype=idx_w_c.dtype, device=idx_w_c.device), idx_w_c)
                        except Exception:
                            pass

                    wk_c = self.mem_wkey(u_c)     # (B,C,key_dim)
                    v_c = self.mem_value(u_c)     # (B,C,mem_dim)

                    for h in range(Hh):
                        write_buckets_h: list[Tensor] = [idx_w_c[:, :, h].to(dtype=torch.long)]
                        if (
                            self.mem_router == "vq"
                            and bool(getattr(self, "mem_write_multi", False))
                            and isinstance(cand_w_idx_all, Tensor)
                            and isinstance(cand_w_w_all, Tensor)
                        ):
                            try:
                                cw = cand_w_w_all[:, t0:t1, h, :]  # (B,C,Nc)
                                ci = cand_w_idx_all[:, t0:t1, h, :]  # (B,C,Nc)
                                k2 = min(2, int(cw.size(-1)))
                                if k2 >= 2:
                                    top2 = torch.topk(cw, k=int(k2), dim=-1).indices  # (B,C,2)
                                    b1 = torch.gather(ci, dim=-1, index=top2[:, :, 1:2]).squeeze(-1)  # (B,C)
                                    write_buckets_h.append(b1.to(dtype=torch.long))
                            except Exception:
                                pass

                        for bidx in write_buckets_h:
                            mk = mem_k[:, h, :, :, :]   # (B,buckets,A,key_dim)
                            mv = mem_v[:, h, :, :, :]   # (B,buckets,A,mem_dim)
                            ml = mem_last[:, h, :, :]   # (B,buckets,A)

                            # Gather along bucket dim=1; index tensor must be 4D like the output:
                            #   mk: (B, buckets, A, key_dim) -> (B, C, A, key_dim)
                            # so create indices shaped (B, C, A, key_dim).
                            idxk = bidx.unsqueeze(-1).unsqueeze(-1).expand(B, C, A, key_dim)
                            idxv = bidx.unsqueeze(-1).unsqueeze(-1).expand(B, C, A, mem_dim)
                            # ml: (B, buckets, A) -> (B, C, A) so index must be 3D (B,C,A)
                            idxl = bidx.unsqueeze(-1).expand(B, C, A)
                            bk_w = mk.gather(dim=1, index=idxk)
                            bv_w = mv.gather(dim=1, index=idxv)
                            bl_w = ml.gather(dim=1, index=idxl)

                            valid_w = bl_w >= 0
                            sim_w = (bk_w * wk_c.unsqueeze(2)).sum(dim=-1) * float(key_scale)
                            sim_w = sim_w.masked_fill(~valid_w, float("-inf"))
                            best_slot = sim_w.argmax(dim=-1)
                            best_sim = sim_w.max(dim=-1).values
                            has_empty = (~valid_w).any(dim=-1)
                            first_empty = (~valid_w).to(torch.int64).argmax(dim=-1)
                            lru_slot = bl_w.argmin(dim=-1)
                            repl_slot = torch.where(has_empty, first_empty, lru_slot)
                            use_update = torch.isfinite(best_sim) & (best_sim >= float(match_thr)) & has_empty.logical_not()
                            slot = torch.where(use_update, best_slot, repl_slot).to(dtype=torch.long)

                            bucket_ev = bidx[b_ev_all, t_ev_all]
                            slot_ev = slot[b_ev_all, t_ev_all]
                            eta_ev = w_eta_c[b_ev_all, t_ev_all].to(dtype=x.dtype)
                            wk_ev = wk_c[b_ev_all, t_ev_all, :].to(dtype=x.dtype)
                            v_ev = v_c[b_ev_all, t_ev_all, :].to(dtype=x.dtype)
                            upd_ev = use_update[b_ev_all, t_ev_all]
                            time_ev = (step0 + t0) + t_ev_all.to(torch.long)

                            key_ev = (((b_ev_all.to(torch.long) * Hh + int(h)) * int(self.mem_buckets) + bucket_ev) * A + slot_ev)
                            winner = last_write_wins(key_ev, time_ev, big=big)

                            bw = b_ev_all[winner]
                            buckw = bucket_ev[winner]
                            slotw = slot_ev[winner]
                            etaw = eta_ev[winner].view(-1, 1)
                            wkw = wk_ev[winner]
                            vw = v_ev[winner]
                            updw = upd_ev[winner].view(-1, 1)
                            timew = time_ev[winner]

                            curk = mem_k[bw, h, buckw, slotw, :]
                            curv = mem_v[bw, h, buckw, slotw, :]
                            # Writes are state updates, not part of the grad graph.
                            with torch.no_grad():
                                mem_k[bw, h, buckw, slotw, :] = torch.where(
                                    updw, (1.0 - etaw) * curk + etaw * wkw, wkw
                                )
                                mem_v[bw, h, buckw, slotw, :] = torch.where(
                                    updw, (1.0 - etaw) * curv + etaw * vw, vw
                                )
                                mem_last[bw, h, buckw, slotw] = timew

            st.step = int(step0 + int(T))
            return _finalize()

        for t in range(int(T)):
            ut = u[:, t, :]  # (B, D)
            # Scheduled sampling mask: whether to apply teacher controls at this position.
            use_teacher_mask = None
            if teacher is not None and (teacher_p < 1.0 or teacher_p > 0.0):
                if teacher_p <= 0.0:
                    use_teacher_mask = torch.zeros((B,), device=x.device, dtype=torch.bool)
                elif teacher_p >= 1.0 or (not self.training):
                    use_teacher_mask = torch.ones((B,), device=x.device, dtype=torch.bool)
                else:
                    use_teacher_mask = (torch.rand((B,), device=x.device) < float(teacher_p))

            # --- State bank update ---
            inp = self.state_in(ut).view(B, self.state_k, D)
            s = decay * s + inp
            g_t = self.state_out(s.reshape(B, self.state_k * D))
            g_seq[:, t, :] = g_t

            # Utility prediction head (supervisable in curriculum).
            util_logit = self.mem_utility_head(ut).squeeze(-1)
            if collect_aux:
                util_logits_seq[:, t] = util_logit

            # Opcode head (logging only; semantics are layered later).
            if self.opcodes_enabled and self.opcode_head is not None and collect_aux and opcode_logits_seq is not None:
                opcode_logits_seq[:, t, :] = self.opcode_head(ut).to(dtype=torch.float32)

            # --- Register read/write (optional, strict sequential) ---
            if self.reg_slots > 0 and isinstance(regs, Tensor):
                assert self.reg_write_gate is not None and self.reg_sel is not None and self.reg_value is not None
                # Read: soft slot selection (constant-time; slots are small).
                sim_r = torch.einsum("brd,bd->br", regs, ut) * (1.0 / math.sqrt(float(D)))
                w_r = torch.softmax(sim_r.float(), dim=-1).to(dtype=ut.dtype)
                z_t = (w_r.unsqueeze(-1) * regs).sum(dim=1)
                if z_seq is not None:
                    z_seq[:, t, :] = z_t

                # Write: gated, top-1 slot select + overwrite/blend.
                reg_gate_logit = self.reg_write_gate(ut).squeeze(-1)  # (B,)
                sel_logits = self.reg_sel(ut)  # (B,R)
                if collect_aux:
                    if reg_write_gate_logits_seq is not None:
                        reg_write_gate_logits_seq[:, t] = reg_gate_logit
                    if reg_sel_logits_seq is not None:
                        reg_sel_logits_seq[:, t, :] = sel_logits.to(dtype=torch.float32)
                p_wr = torch.sigmoid(reg_gate_logit)
                thr_r = float(getattr(self.config, "reg_write_threshold", 0.5))
                eta_r = float(getattr(self.config, "reg_write_eta", 1.0))
                do_wr = p_wr > float(thr_r)
                if bool(do_wr.any().item()):
                    slot = sel_logits.argmax(dim=-1)  # (B,)
                    val = self.reg_value(ut)  # (B,D)
                    b = batch_idx[do_wr]
                    s_idx = slot[do_wr].to(dtype=torch.long)
                    v_wr = val[do_wr]
                    if eta_r >= 1.0 - 1e-9:
                        regs = regs.clone()
                        regs[b, s_idx, :] = v_wr
                    else:
                        regs = regs.clone()
                        cur = regs[b, s_idx, :]
                        regs[b, s_idx, :] = (1.0 - float(eta_r)) * cur + float(eta_r) * v_wr

            # Router + address selection (bits or VQ).
            cand_r_idx = cand_r_w = None
            cand_w_idx = cand_w_w = None
            if self.mem_router == "bits":
                assert self.mem_read_bits is not None and self.mem_write_bits is not None
                z_read = self.mem_read_bits(ut).view(B, self.mem_hashes, self.mem_bits)
                z_write = self.mem_write_bits(ut).view(B, self.mem_hashes, self.mem_bits)
                if collect_aux and read_bit_logits_seq is not None and write_bit_logits_seq is not None:
                    read_bit_logits_seq[:, t, :, :] = z_read
                    write_bit_logits_seq[:, t, :, :] = z_write
                idx_r = self._route_bits(z_read)
                idx_w = self._route_bits(z_write)
            else:
                assert self.mem_vq_proj_r is not None and self.mem_vq_proj_w is not None
                assert self.mem_vq_codebook_r is not None and self.mem_vq_codebook_w is not None
                G = int(self.mem_vq_groups)
                gd = int(self.mem_vq_group_dim)
                route_dim = int(G * gd)
                yr = self.mem_vq_proj_r(ut).view(B, self.mem_hashes, route_dim).view(B, self.mem_hashes, G, gd)
                yw = self.mem_vq_proj_w(ut).view(B, self.mem_hashes, route_dim).view(B, self.mem_hashes, G, gd)
                idx_r, cand_r_idx, cand_r_w, gl_r = self._route_vq(
                    yr, codebook=self.mem_vq_codebook_r, return_group_logits=bool(collect_aux)
                )
                idx_w, cand_w_idx, cand_w_w, gl_w = self._route_vq(
                    yw, codebook=self.mem_vq_codebook_w, return_group_logits=bool(collect_aux)
                )
                if collect_aux and isinstance(read_vq_logits_seq, Tensor) and isinstance(write_vq_logits_seq, Tensor):
                    if isinstance(gl_r, Tensor):
                        read_vq_logits_seq[:, t, :, :, :] = gl_r
                    if isinstance(gl_w, Tensor):
                        write_vq_logits_seq[:, t, :, :, :] = gl_w
                if collect_aux and read_codes_seq is not None and write_codes_seq is not None:
                    # Store top-1 codes per group for introspection/training.
                    # Recover best codes by recomputing argmin from route_vq inputs.
                    # (Small constant G, K; acceptable overhead.)
                    # Read codes
                    e = self.mem_vq_codebook_r  # (H,G,K,gd)
                    y2 = (yr * yr).sum(dim=-1, keepdim=True)
                    e2 = (e * e).sum(dim=-1).view(1, self.mem_hashes, G, int(self.mem_vq_codebook_size))
                    dot = torch.einsum("bhgd,hgkd->bhgk", yr, e)
                    dist = y2 + e2 - 2.0 * dot
                    best_codes = dist.argmin(dim=-1)  # (B,H,G)
                    read_codes_seq[:, t, :, :] = best_codes
                    # Write codes
                    e2w = (self.mem_vq_codebook_w * self.mem_vq_codebook_w).sum(dim=-1).view(1, self.mem_hashes, G, int(self.mem_vq_codebook_size))
                    y2w = (yw * yw).sum(dim=-1, keepdim=True)
                    dotw = torch.einsum("bhgd,hgkd->bhgk", yw, self.mem_vq_codebook_w)
                    distw = y2w + e2w - 2.0 * dotw
                    bestw = distw.argmin(dim=-1)
                    write_codes_seq[:, t, :, :] = bestw

            # --- Hash memory read (set-associative, fuzzy within bucket) ---
            # Teacher override (bucket indices), -1 means "no override".
            if teacher is not None and "read_bucket" in teacher:
                tb = teacher["read_bucket"]
                if isinstance(tb, Tensor):
                    # Accept (B,T), (B,T,H), (T,H), (T,) shapes.
                    try:
                        if tb.dim() == 2 and tb.size(0) == B and tb.size(1) == T:
                            tb_t = tb[:, t].view(B, 1).expand(B, self.mem_hashes)
                        elif tb.dim() == 3 and tb.size(0) == B and tb.size(1) == T:
                            tb_t = tb[:, t, :].view(B, self.mem_hashes)
                        elif tb.dim() == 2 and tb.size(0) == T:
                            tb_t = tb[t, :].view(1, self.mem_hashes).expand(B, self.mem_hashes)
                        elif tb.dim() == 1 and tb.size(0) == T:
                            tb_t = tb[t].view(1, 1).expand(B, self.mem_hashes)
                        else:
                            tb_t = None
                        if tb_t is not None:
                            use = tb_t >= 0
                            if isinstance(use_teacher_mask, Tensor):
                                use = use & use_teacher_mask.view(B, 1).expand_as(use)
                            idx_r = torch.where(use, tb_t.to(dtype=idx_r.dtype, device=idx_r.device), idx_r)
                    except Exception:
                        pass
            qk = self.mem_qkey(ut)  # (B, key_dim)
            # Candidate-bucket reads (for VQ beam) remain O(1) since Ncand is constant.
            if cand_r_idx is None or cand_r_w is None:
                # Single bucket read.
                bk = mem_k[batch_idx.view(B, 1), hash_idx, idx_r, :, :]  # (B,H,A,key_dim)
                bv = mem_v[batch_idx.view(B, 1), hash_idx, idx_r, :, :]  # (B,H,A,mem_dim)
                bl = mem_last[batch_idx.view(B, 1), hash_idx, idx_r, :]  # (B,H,A)
                valid = bl >= 0
                sim = (bk * qk.view(B, 1, 1, self.mem_key_dim)).sum(dim=-1) * float(key_scale)  # (B,H,A)
                sim = sim.masked_fill(~valid, float("-inf"))
                any_valid = valid.any(dim=-1, keepdim=True)  # (B,H,1)
                w = torch.softmax(sim / float(temp), dim=-1)
                w = torch.where(any_valid, w, torch.zeros_like(w))
                read_h = (w.unsqueeze(-1) * bv).sum(dim=2)  # (B,H,mem_dim)
            else:
                # Multi-bucket read: cand_r_idx is (B,H,Nc), cand_r_w is (B,H,Nc)
                Nc = int(cand_r_idx.size(-1))
                # Normalize candidate weights.
                ww = cand_r_w / (cand_r_w.sum(dim=-1, keepdim=True).clamp_min(1e-9))
                read_h = ut.new_zeros((B, self.mem_hashes, self.mem_dim))
                for ci in range(Nc):
                    bidx = cand_r_idx[:, :, ci]  # (B,H)
                    bk = mem_k[batch_idx.view(B, 1), hash_idx, bidx, :, :]
                    bv = mem_v[batch_idx.view(B, 1), hash_idx, bidx, :, :]
                    bl = mem_last[batch_idx.view(B, 1), hash_idx, bidx, :]
                    valid = bl >= 0
                    sim = (bk * qk.view(B, 1, 1, self.mem_key_dim)).sum(dim=-1) * float(key_scale)
                    sim = sim.masked_fill(~valid, float("-inf"))
                    any_valid = valid.any(dim=-1, keepdim=True)
                    wslot = torch.softmax(sim / float(temp), dim=-1)
                    wslot = torch.where(any_valid, wslot, torch.zeros_like(wslot))
                    rh_ci = (wslot.unsqueeze(-1) * bv).sum(dim=2)  # (B,H,mem_dim)
                    read_h = read_h + rh_ci * ww[:, :, ci].unsqueeze(-1)
            read_sum = read_h.sum(dim=1)  # (B, mem_dim)
            r_t = self.mem_out(read_sum)
            r_seq[:, t, :] = r_t

            # --- Hash memory write (sparse/evented) ---
            gate_logit = self.mem_write_gate(ut).squeeze(-1)  # (B,)
            if collect_aux:
                gate_logits_seq[:, t] = gate_logit
            p = torch.sigmoid(gate_logit)  # (B,)
            # Default: thresholded learned gate.
            mask = (p > thr).to(dtype=ut.dtype)  # (B,)
            # Teacher override: write_gate (0/1, or -1 to ignore).
            if teacher is not None and "write_gate" in teacher:
                tg = teacher["write_gate"]
                if isinstance(tg, Tensor):
                    try:
                        if tg.dim() == 2 and tg.size(0) == B and tg.size(1) == T:
                            tg_t = tg[:, t]
                        elif tg.dim() == 1 and tg.size(0) == T:
                            tg_t = tg[t].view(1).expand(B)
                        else:
                            tg_t = None
                        if tg_t is not None:
                            use = tg_t >= 0
                            if isinstance(use_teacher_mask, Tensor):
                                use = use & use_teacher_mask
                            mask = torch.where(
                                use,
                                (tg_t > 0).to(dtype=mask.dtype, device=mask.device),
                                mask,
                            )
                    except Exception:
                        pass
            if bool((mask > 0).any().item()):
                # Teacher override for write buckets.
                if teacher is not None and "write_bucket" in teacher:
                    tbw = teacher["write_bucket"]
                    if isinstance(tbw, Tensor):
                        try:
                            if tbw.dim() == 2 and tbw.size(0) == B and tbw.size(1) == T:
                                tbw_t = tbw[:, t].view(B, 1).expand(B, self.mem_hashes)
                            elif tbw.dim() == 3 and tbw.size(0) == B and tbw.size(1) == T:
                                tbw_t = tbw[:, t, :].view(B, self.mem_hashes)
                            elif tbw.dim() == 2 and tbw.size(0) == T:
                                tbw_t = tbw[t, :].view(1, self.mem_hashes).expand(B, self.mem_hashes)
                            elif tbw.dim() == 1 and tbw.size(0) == T:
                                tbw_t = tbw[t].view(1, 1).expand(B, self.mem_hashes)
                            else:
                                tbw_t = None
                            if tbw_t is not None:
                                use = tbw_t >= 0
                                if isinstance(use_teacher_mask, Tensor):
                                    use = use & use_teacher_mask.view(B, 1).expand_as(use)
                                idx_w = torch.where(use, tbw_t.to(dtype=idx_w.dtype, device=idx_w.device), idx_w)
                        except Exception:
                            pass
                wk = self.mem_wkey(ut)  # (B, key_dim)
                v = self.mem_value(ut)  # (B, mem_dim)
                # Effective eta can be scaled by p to make writes smooth.
                w_eta = (eta * p).to(dtype=ut.dtype) * mask
                # Write to one (or more) buckets.
                write_buckets: list[Tensor] = [idx_w]
                if self.mem_router == "vq" and self.mem_write_multi and isinstance(cand_w_idx, Tensor) and isinstance(cand_w_w, Tensor):
                    # Use top-2 candidate buckets by weight (constant factor).
                    try:
                        k2 = min(2, int(cand_w_w.size(-1)))
                        if k2 >= 2:
                            top2 = torch.topk(cand_w_w, k=int(k2), dim=-1).indices  # (B,H,2)
                            b1 = torch.gather(cand_w_idx, dim=-1, index=top2[:, :, 1:2]).squeeze(-1)
                            write_buckets.append(b1)
                    except Exception as e:
                        logger.warning(f"Error in write_buckets: {e}")

                for idx_w_use in write_buckets:
                    wk_b = wk.view(B, 1, 1, self.mem_key_dim)
                    bk_w = mem_k[batch_idx.view(B, 1), hash_idx, idx_w_use, :, :]  # (B,H,A,key_dim)
                    bv_w = mem_v[batch_idx.view(B, 1), hash_idx, idx_w_use, :, :]  # (B,H,A,mem_dim)
                    bl_w = mem_last[batch_idx.view(B, 1), hash_idx, idx_w_use, :]  # (B,H,A)
                    valid_w = bl_w >= 0
                    sim_w = (bk_w * wk_b).sum(dim=-1) * float(key_scale)  # (B,H,A)
                    sim_w = sim_w.masked_fill(~valid_w, float("-inf"))
                    best_slot = sim_w.argmax(dim=-1)  # (B,H)
                    best_sim = sim_w.max(dim=-1).values  # (B,H)

                    has_empty = (~valid_w).any(dim=-1)  # (B,H)
                    first_empty = (~valid_w).to(torch.int64).argmax(dim=-1)
                    lru_slot = bl_w.argmin(dim=-1)
                    repl_slot = torch.where(has_empty, first_empty, lru_slot)

                    use_update = torch.isfinite(best_sim) & (best_sim >= float(match_thr)) & has_empty.logical_not()

                    for h in range(self.mem_hashes):
                        a = w_eta.view(B, 1)
                        b = batch_idx
                        bh = torch.full((B,), h, device=x.device, dtype=torch.long)
                        bucket = idx_w_use[:, h]
                        slot_u = best_slot[:, h]
                        slot_r = repl_slot[:, h]
                        do_u = use_update[:, h] & (w_eta > 0)
                        do_r = (~use_update[:, h]) & (w_eta > 0)

                        if bool(do_u.any().item()):
                            bu = b[do_u]
                            bkh = bh[do_u]
                            bb = bucket[do_u]
                            ss = slot_u[do_u]
                            aa = a[do_u]
                            curk = mem_k[bu, bkh, bb, ss, :]
                            curv = mem_v[bu, bkh, bb, ss, :]
                            with torch.no_grad():
                                mem_k[bu, bkh, bb, ss, :] = (1.0 - aa) * curk + aa * wk[do_u]
                                mem_v[bu, bkh, bb, ss, :] = (1.0 - aa) * curv + aa * v[do_u]
                                mem_last[bu, bkh, bb, ss] = int(st.step)

                        if bool(do_r.any().item()):
                            br = b[do_r]
                            bkh = bh[do_r]
                            bb = bucket[do_r]
                            ss = slot_r[do_r]
                            with torch.no_grad():
                                mem_k[br, bkh, bb, ss, :] = wk[do_r]
                                mem_v[br, bkh, bb, ss, :] = v[do_r]
                                mem_last[br, bkh, bb, ss] = int(st.step)

            st.step += 1

        return _finalize()

