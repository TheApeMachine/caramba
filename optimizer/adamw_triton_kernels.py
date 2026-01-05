"""CUDA/Triton AdamW master-step kernels.

Implements the same update semantics as the Metal `adamw_master_step_fp16` kernel:
- Parameters: fp16/bf16 (updated in-place)
- Gradients:  fp16/bf16
- State:      fp32 master weights + fp32 exp_avg + fp32 exp_avg_sq (updated)

The kernel performs decoupled weight decay on the fp32 master weights:
  w = w * (1 - lr_wd)
then updates:
  m = beta1*m + (1-beta1)*g
  v = beta2*v + (1-beta2)*g^2
  w = w - step_size * m / (sqrt(v) + eps)
and finally writes p = cast(w) to fp16/bf16.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from caramba.optimizer.triton_runtime import TRITON_AVAILABLE


if not TYPE_CHECKING and TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    @triton.jit
    def adamw_master_step(
        p_ptr,
        g_ptr,
        master_ptr,
        m_ptr,
        v_ptr,
        n_elements: tl.constexpr,
        step_size: tl.constexpr,
        beta1: tl.constexpr,
        beta2: tl.constexpr,
        eps: tl.constexpr,
        lr_wd: tl.constexpr,
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
        if lr_wd != 0.0:
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

else:
    adamw_master_step = None  # type: ignore[assignment]

