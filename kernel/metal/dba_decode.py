"""Fused DBA decode wrapper for the Metal extension."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from caramba.optimizer.runtime import metal_supported

from .jit import load_caramba_metal_ops

if TYPE_CHECKING:
    from torch import Tensor


def metal_dba_decode_available() -> bool:
    """Whether the runtime is capable of using the Metal DBA decode path.

    Note: this answers "can we run/build it" (Darwin + MPS). The actual extension
    may still fail to build depending on the local toolchain.
    """
    return metal_supported()


def _squeeze_q(q: Tensor) -> Tensor:
    if q.dim() == 4:
        if q.size(2) != 1:
            raise ValueError("decode expects q.shape[2] == 1")
        return q[:, :, 0, :]
    if q.dim() == 3:
        return q
    raise ValueError(f"expected q with 3 or 4 dims, got shape={tuple(q.shape)}")


def dba_decode_fp16(
    *,
    q_sem: Tensor,
    q_geo: Tensor,
    k_sem: Tensor,
    k_geo: Tensor,
    v: Tensor,
    k_sem_null: Tensor | None = None,
    k_geo_null: Tensor | None = None,
    v_null: Tensor | None = None,
    sem_scale: float | None = None,
    geo_scale: float | None = None,
    verbose_build: bool = False,
) -> Tensor:
    """Fused DBA decode (MPS/Metal) for fp16 KV caches.

    Shapes (contiguous required):
      - q_sem: (B, H, 1, sem_hd) or (B, H, sem_hd)
      - q_geo: (B, H, 1, geo_hd) or (B, H, geo_hd)
      - k_sem: (B, S, H*sem_hd)
      - k_geo: (B, S, H*geo_hd)
      - v:     (B, S, H*v_hd)

    Returns:
      - out: (B, H, 1, v_hd) in fp16
    """
    if q_sem.device.type != "mps":
        raise RuntimeError("Metal DBA decode requires device.type == 'mps'")

    q_sem2 = _squeeze_q(q_sem).contiguous().to(torch.float16)
    q_geo2 = _squeeze_q(q_geo).contiguous().to(torch.float16)
    # K/V are usually views into a larger preallocated cache buffer. Do NOT
    # force contiguity here unless required (it would copy).
    k_sem2 = k_sem.to(torch.float16)
    k_geo2 = k_geo.to(torch.float16)
    v2 = v.to(torch.float16)

    # Kernel expects last dim contiguous; if not, make it contiguous (rare).
    if k_sem2.stride(-1) != 1:
        k_sem2 = k_sem2.contiguous()
    if k_geo2.stride(-1) != 1:
        k_geo2 = k_geo2.contiguous()
    if v2.stride(-1) != 1:
        v2 = v2.contiguous()

    use_null = k_sem_null is not None or k_geo_null is not None or v_null is not None
    if use_null:
        if k_sem_null is None or k_geo_null is None or v_null is None:
            raise ValueError("k_sem_null, k_geo_null, and v_null must be provided together")

        ksn = _squeeze_q(k_sem_null).contiguous().to(torch.float16)  # type: ignore[arg-type]
        kgn = _squeeze_q(k_geo_null).contiguous().to(torch.float16)  # type: ignore[arg-type]
        vn = _squeeze_q(v_null).contiguous().to(torch.float16)  # type: ignore[arg-type]
    else:
        ksn = kgn = vn = None

    if q_sem2.dim() != 3 or q_geo2.dim() != 3:
        raise RuntimeError("q tensors must be (B,H,D)")
    if k_sem2.dim() != 3 or k_geo2.dim() != 3 or v2.dim() != 3:
        raise RuntimeError("k/v tensors must be (B,S,D)")

    B, H, sem_hd = q_sem2.shape
    B2, H2, geo_hd = q_geo2.shape
    if B2 != B or H2 != H:
        raise ValueError("q_sem and q_geo must agree on (B,H)")

    Bk, S, sem_dim = k_sem2.shape
    Bg, S2, geo_dim = k_geo2.shape
    Bv, S3, v_dim = v2.shape
    if not (Bk == Bg == Bv == B):
        raise ValueError("k/v batch size mismatch")
    if not (S == S2 == S3):
        raise ValueError("k/v sequence length mismatch")
    if sem_dim != H * sem_hd:
        raise ValueError(f"k_sem last dim must be H*sem_hd ({H*sem_hd}), got {sem_dim}")
    if geo_dim != H * geo_hd:
        raise ValueError(f"k_geo last dim must be H*geo_hd ({H*geo_hd}), got {geo_dim}")
    if v_dim % H != 0:
        raise ValueError("v last dim must be divisible by H")

    v_hd = v_dim // H
    if v_hd > 256:
        raise ValueError(f"Metal DBA decode requires v_hd <= 256 (got v_hd={v_hd})")
    if sem_scale is None:
        sem_scale = 1.0 / math.sqrt(float(sem_hd))
    if geo_scale is None:
        geo_scale = 1.0 / math.sqrt(float(geo_hd))

    ops = load_caramba_metal_ops(verbose=bool(verbose_build))

    if use_null:
        assert ksn is not None, "ksn is None"
        assert kgn is not None, "kgn is None"
        assert vn is not None, "vn is None"
        if ksn.shape != (B, H, sem_hd):
            raise ValueError(f"k_sem_null must be (B,H,sem_hd) == {(B, H, sem_hd)}, got {tuple(ksn.shape)}")
        if kgn.shape != (B, H, geo_hd):
            raise ValueError(f"k_geo_null must be (B,H,geo_hd) == {(B, H, geo_hd)}, got {tuple(kgn.shape)}")
        if vn.shape != (B, H, v_hd):
            raise ValueError(f"v_null must be (B,H,v_hd) == {(B, H, v_hd)}, got {tuple(vn.shape)}")
        out = ops.dba_decode_null(
            q_sem2,
            k_sem2,
            q_geo2,
            k_geo2,
            v2,
            ksn,
            kgn,
            vn,
            float(sem_scale),
            float(geo_scale),
        )
    else:
        out = ops.dba_decode(
            q_sem2,
            k_sem2,
            q_geo2,
            k_geo2,
            v2,
            float(sem_scale),
            float(geo_scale),
        )
    # Extension returns (B,H,v_hd).
    if out.shape != (B, H, v_hd):
        raise RuntimeError(f"unexpected output shape {tuple(out.shape)}")
    return out.unsqueeze(2)
