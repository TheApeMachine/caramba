"""CUDA/Triton AdamW master-step kernels.

Implements the same update semantics as the Metal `adamw_master_step_fp16` kernel:
- Parameters: fp16/bf16 (updated in-place)
- Gradients:  fp16/bf16
- State:      fp32 master weights + fp32 exp_avg + fp32 exp_avg_sq (updated)

Important: Only true compile-time knobs should be marked `tl.constexpr`.
Per-step scalars like `step_size` must remain runtime arguments to avoid
specializing/compiling a new kernel variant every optimizer step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from caramba.console import logger
from caramba.optimizer.triton_runtime import TRITON_AVAILABLE

__all__ = ["adamw_master_step"]

# Placeholder so imports succeed even when Triton isn't available.
adamw_master_step: Any | None = None


if not TYPE_CHECKING and TRITON_AVAILABLE:
    try:  # pyright: ignore[reportUnreachable]
        import triton
        import triton.language as tl
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"Failed to import Triton, continuing: {e}")
    else:

        @triton.jit
        def adamw_master_step(
            p_ptr,
            g_ptr,
            master_ptr,
            m_ptr,
            v_ptr,
            # IMPORTANT: runtime scalars (NOT tl.constexpr)
            n_elements,
            step_size,
            beta1,
            beta2,
            eps,
            lr_wd,
            # Only keep true compile-time knobs as constexpr
            USE_BF16: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n_elements

            # Load grad + state. Grad/param are fp16 or bf16; compute in fp32.
            g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(master_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            v = tl.load(v_ptr + offs, mask=mask, other=0.0).to(tl.float32)

            # Decoupled weight decay (AdamW): apply to master weights.
            # Always applying this is correct even when lr_wd == 0.0.
            w = w * (1.0 - lr_wd)

            # Moments.
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g * g)

            denom = tl.sqrt(v) + eps
            w = w - step_size * (m / denom)

            # Store updated state.
            tl.store(m_ptr + offs, m, mask=mask)
            tl.store(v_ptr + offs, v, mask=mask)
            tl.store(master_ptr + offs, w, mask=mask)

            # Store updated params, matching input dtype.
            out = w.to(tl.bfloat16) if USE_BF16 else w.to(tl.float16)
            tl.store(p_ptr + offs, out, mask=mask)

