"""Resonant phase update Triton kernels.

Implements the elementwise update+normalize step used by the resonant router on CUDA.

This file must be safe to import on non-CUDA platforms (e.g. MPS-only dev), so
kernel definitions are registered only when Triton is available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from console import logger
from optimizer.runtime import triton_supported

__all__ = ["resonant_update_fwd", "resonant_update_bwd"]

resonant_update_fwd: object | None = None
resonant_update_bwd: object | None = None

if not TYPE_CHECKING and triton_supported():
    try:  # pyright: ignore[reportUnreachable]
        import triton
        import triton.language as tl
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"Failed to import Triton, continuing: {e}")
    else:

        @triton.jit
        def resonant_update_fwd(
            x_ptr,
            y_ptr,
            vr_ptr,
            vi_ptr,
            diag_ptr,  # (H, D)
            x_out_ptr,
            y_out_ptr,
            a_ptr,  # saved x_new
            b_ptr,  # saved y_new
            inv_r_ptr,  # saved 1/r
            n_elements: tl.constexpr,
            D: tl.constexpr,
            H: tl.constexpr,
            inv_D: tl.constexpr,
            scale: tl.constexpr,
            damping: tl.constexpr,
            zero_diag: tl.constexpr,
            BLOCK: tl.constexpr = 256,
        ):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            m = offs < n_elements

            x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
            y = tl.load(y_ptr + offs, mask=m, other=0.0).to(tl.float32)
            vr = tl.load(vr_ptr + offs, mask=m, other=0.0).to(tl.float32)
            vi = tl.load(vi_ptr + offs, mask=m, other=0.0).to(tl.float32)

            d = offs % D
            tmp = offs // D
            h = tmp % H

            diag = tl.load(diag_ptr + h * D + d, mask=m, other=0.0).to(tl.float32)
            one_minus = tl.full((BLOCK,), 1.0 - damping, tl.float32)

            cr = vr * inv_D
            ci = vi * inv_D
            if zero_diag:
                cr = cr - diag * x
                ci = ci - diag * y

            a = x * one_minus + scale * cr
            b = y * one_minus + scale * ci
            inv_r = tl.rsqrt(a * a + b * b + 1e-12)
            xo = a * inv_r
            yo = b * inv_r

            tl.store(x_out_ptr + offs, xo, mask=m)
            tl.store(y_out_ptr + offs, yo, mask=m)
            tl.store(a_ptr + offs, a, mask=m)
            tl.store(b_ptr + offs, b, mask=m)
            tl.store(inv_r_ptr + offs, inv_r, mask=m)

        @triton.jit
        def resonant_update_bwd(
            grad_xo_ptr,
            grad_yo_ptr,
            diag_ptr,
            grad_vr_ptr,
            grad_vi_ptr,
            grad_x_ptr,
            grad_y_ptr,
            a_ptr,
            b_ptr,
            inv_r_ptr,
            n_elements: tl.constexpr,
            D: tl.constexpr,
            H: tl.constexpr,
            inv_D: tl.constexpr,
            scale: tl.constexpr,
            damping: tl.constexpr,
            zero_diag: tl.constexpr,
            BLOCK: tl.constexpr = 256,
        ):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            m = offs < n_elements

            gxo = tl.load(grad_xo_ptr + offs, mask=m, other=0.0).to(tl.float32)
            gyo = tl.load(grad_yo_ptr + offs, mask=m, other=0.0).to(tl.float32)
            a = tl.load(a_ptr + offs, mask=m, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + offs, mask=m, other=0.0).to(tl.float32)
            inv_r = tl.load(inv_r_ptr + offs, mask=m, other=0.0).to(tl.float32)

            d = offs % D
            tmp = offs // D
            h = tmp % H
            diag = tl.load(diag_ptr + h * D + d, mask=m, other=0.0).to(tl.float32)

            inv_r2 = inv_r * inv_r
            inv_r3 = inv_r2 * inv_r
            dot = gxo * a + gyo * b
            ga = gxo * inv_r - a * dot * inv_r3
            gb = gyo * inv_r - b * dot * inv_r3

            coeff = tl.full((BLOCK,), 1.0 - damping, tl.float32)
            if zero_diag:
                coeff = coeff - scale * diag

            gx = ga * coeff
            gy = gb * coeff
            gvr = ga * (scale * inv_D)
            gvi = gb * (scale * inv_D)

            tl.store(grad_x_ptr + offs, gx, mask=m)
            tl.store(grad_y_ptr + offs, gy, mask=m)
            tl.store(grad_vr_ptr + offs, gvr, mask=m)
            tl.store(grad_vi_ptr + offs, gvi, mask=m)

        # Export symbols
        resonant_update_fwd = resonant_update_fwd  # type: ignore[assignment]
        resonant_update_bwd = resonant_update_bwd  # type: ignore[assignment]

