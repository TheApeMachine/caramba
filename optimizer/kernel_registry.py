"""Kernel registry and startup validation.

Caramba operates on a strict policy when an accelerator is available:
- Pick the fastest supported kernel path deterministically.
- Validate required kernel backends at startup.
- If a required kernel backend is unavailable, fail loudly with an actionable error.
- Log the chosen performance paths exactly once at initialization.

In CPU-only environments (no CUDA/MPS), the registry initializes in a "no fused
kernels" mode so import-time behavior remains safe and unit tests can run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import platform

import torch

from caramba.console import logger
from caramba.optimizer.runtime import (
    metal_build_tools_available,
    metal_supported,
    triton_supported,
)


@dataclass(frozen=True, slots=True)
class KernelRegistry:
    cuda_available: bool
    mps_available: bool
    triton_available: bool
    metal_supported: bool
    metal_build_tools_available: bool
    metal_ops_loaded: bool


_REGISTRY: KernelRegistry | None = None
_LOGGED: bool = False


def _cuda_device_summary() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    try:
        idx = int(torch.cuda.current_device())
        name = str(torch.cuda.get_device_name(idx))
        cap = ".".join(str(x) for x in torch.cuda.get_device_capability(idx))
        return f"{name} (sm_{cap})"
    except Exception as e:
        logger.warning(f"CUDA available but device query failed: {e!r}")
        return "CUDA available (device query failed)"


def _require(condition: bool, *, msg: str) -> None:
    if not condition:
        raise RuntimeError(msg)


def initialize_kernels() -> KernelRegistry:
    """Initialize and validate accelerator kernel backends.

    This must run exactly once per process (idempotent).
    """
    global _REGISTRY, _LOGGED
    if _REGISTRY is not None:
        return _REGISTRY

    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(torch.backends.mps.is_available())
    # CPU-only mode: no validation, no fused kernels.
    if not cuda_available and not mps_available:
        _REGISTRY = KernelRegistry(
            cuda_available=False,
            mps_available=False,
            triton_available=False,
            metal_supported=bool(metal_supported()),
            metal_build_tools_available=bool(metal_build_tools_available()),
            metal_ops_loaded=False,
        )
        return _REGISTRY

    # ---- Metal/MPS validation (compile+load extension at startup) ----
    metal_ops_loaded = False
    if mps_available:
        _require(
            bool(metal_supported()),
            msg=(
                "MPS is available but Metal is marked unsupported by caramba.\n"
                f"platform.system()={platform.system()!r}\n"
            ),
        )
        _require(
            bool(metal_build_tools_available()),
            msg=(
                "Metal/MPS is available but the Metal build toolchain is not.\n"
                "Install Xcode Command Line Tools and ensure `xcrun -sdk macosx --find metal` works.\n"
            ),
        )
        try:
            from caramba.optimizer.metal.jit import load_caramba_metal_ops

            _ = load_caramba_metal_ops(verbose=False)
            from caramba.optimizer.metal.attention_jit import load_caramba_metal_attention_ops

            _ = load_caramba_metal_attention_ops(verbose=False)
            metal_ops_loaded = True
        except Exception as e:
            raise RuntimeError(
                "Metal kernel compilation/loading failed.\n"
                "This is a hard failure under the kernel policy.\n"
                f"Error: {type(e).__name__}: {e}\n"
            ) from e

    # ---- CUDA/Triton validation ----
    if cuda_available:
        _require(
            bool(triton_supported()),
            msg=(
                "CUDA is available but Triton is not.\n"
                "Install Triton (and its CUDA dependencies) so CUDA fused kernels can be used.\n"
                f"CUDA device: {_cuda_device_summary()}\n"
            ),
        )
        # Validate that required Triton decode kernels are importable/defined.
        from caramba.optimizer.kernels_decoupled import (
            kv_decode_partition_stats_decoupled_q4q8q4,
            kv_decode_reduce_partitions,
            kv_decode_update_decoupled_q4q8q4,
        )
        from caramba.optimizer.flash_attention_triton_kernels_bwd import (
            flash_attn_bwd_dkv,
            flash_attn_bwd_dq,
            flash_attn_bwd_preprocess,
        )
        from caramba.optimizer.flash_attention_triton_kernels_fwd import flash_attn_fwd
        from caramba.optimizer.dba_attention_triton_kernels_bwd import (
            dba_attn_bwd_dkv,
            dba_attn_bwd_dq,
            dba_attn_bwd_preprocess,
        )
        from caramba.optimizer.dba_attention_triton_kernels_fwd import dba_attn_fwd
        from caramba.optimizer.kernels_ssm import selective_scan_triton
        from caramba.optimizer.rmsnorm_triton_kernels import rmsnorm_fwd, rmsnorm_bwd_x, rmsnorm_bwd_x_noweight, rmsnorm_bwd_w
        from caramba.optimizer.layernorm_triton_kernels import layernorm_fwd, layernorm_bwd_x, layernorm_gradw, layernorm_gradb
        from caramba.optimizer.rope_triton_kernels import rope_fwd, rope_bwd
        from caramba.optimizer.adamw_triton_kernels import adamw_master_step

        missing = [
            name
            for name, k in [
                ("kv_decode_update_decoupled_q4q8q4", kv_decode_update_decoupled_q4q8q4),
                ("kv_decode_partition_stats_decoupled_q4q8q4", kv_decode_partition_stats_decoupled_q4q8q4),
                ("kv_decode_reduce_partitions", kv_decode_reduce_partitions),
                ("flash_attn_fwd", flash_attn_fwd),
                ("flash_attn_bwd_preprocess", flash_attn_bwd_preprocess),
                ("flash_attn_bwd_dkv", flash_attn_bwd_dkv),
                ("flash_attn_bwd_dq", flash_attn_bwd_dq),
                ("dba_attn_fwd", dba_attn_fwd),
                ("dba_attn_bwd_preprocess", dba_attn_bwd_preprocess),
                ("dba_attn_bwd_dkv", dba_attn_bwd_dkv),
                ("dba_attn_bwd_dq", dba_attn_bwd_dq),
                ("selective_scan_triton", selective_scan_triton),
                ("rmsnorm_fwd", rmsnorm_fwd),
                ("rmsnorm_bwd_x", rmsnorm_bwd_x),
                ("rmsnorm_bwd_x_noweight", rmsnorm_bwd_x_noweight),
                ("rmsnorm_bwd_w", rmsnorm_bwd_w),
                ("layernorm_fwd", layernorm_fwd),
                ("layernorm_bwd_x", layernorm_bwd_x),
                ("layernorm_gradw", layernorm_gradw),
                ("layernorm_gradb", layernorm_gradb),
                ("rope_fwd", rope_fwd),
                ("rope_bwd", rope_bwd),
                ("adamw_master_step", adamw_master_step),
            ]
            if k is None
        ]
        _require(
            not missing,
            msg=(
                "CUDA is available but required Triton kernels are missing.\n"
                f"Missing: {', '.join(missing)}\n"
                "This usually indicates Triton import/JIT issues.\n"
                f"CUDA device: {_cuda_device_summary()}\n"
            ),
        )

    _REGISTRY = KernelRegistry(
        cuda_available=cuda_available,
        mps_available=mps_available,
        triton_available=bool(triton_supported()),
        metal_supported=bool(metal_supported()),
        metal_build_tools_available=bool(metal_build_tools_available()),
        metal_ops_loaded=bool(metal_ops_loaded),
    )

    # ---- One-time path logging ----
    if not _LOGGED:
        _LOGGED = True
        if _REGISTRY.mps_available:
            logger.info("[KERNEL] RMSNorm: Metal fp16 + backward (MPS)")
            logger.info("[KERNEL] LayerNorm: Metal fp16 + backward (MPS)")
            logger.info("[KERNEL] RoPE: Metal fp16 + backward (MPS)")
            logger.info("[KERNEL] Attention Decode: Metal fp16 (MPS, inference)")
            logger.info("[KERNEL] Attention Train: Metal fp16 + backward (MPS)")
            logger.info("[KERNEL] SSM Scan: Metal fp16 + backward (MPS)")
            logger.info("[KERNEL] AdamW Master Step: Metal fused (MPS)")
            logger.info("[KERNEL] Lion Step: Metal fused (MPS)")
        if _REGISTRY.cuda_available:
            logger.info(f"[KERNEL] CUDA device: {_cuda_device_summary()}")
            logger.info("[KERNEL] Attention Decode: Triton q4/q8/q4 decode + split-K (CUDA, inference)")
            logger.info("[KERNEL] Attention Train: Triton FlashAttention forward+backward (CUDA)")
            logger.info("[KERNEL] DBA Attention Train: Triton decoupled FlashAttention forward+backward (CUDA)")
            logger.info("[KERNEL] SSM Scan: Triton selective scan forward+backward (CUDA)")
            logger.info("[KERNEL] RMSNorm: Triton fused forward+backward (CUDA)")
            logger.info("[KERNEL] LayerNorm: Triton fused forward+backward (CUDA)")
            logger.info("[KERNEL] RoPE: Triton forward+backward (CUDA)")
            logger.info("[KERNEL] AdamW Master Step: Triton fused (CUDA)")

    return _REGISTRY


KERNELS: Final[KernelRegistry] = initialize_kernels()

