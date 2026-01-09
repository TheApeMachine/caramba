"""Triton CUDA kernels for LayerNorm."""

from __future__ import annotations

from typing import TYPE_CHECKING

from caramba.console import logger
from caramba.optimizer.runtime import triton_supported

__all__ = [
    "layernorm_fwd",
    "layernorm_bwd_x",
    "layernorm_gradw",
    "layernorm_gradb",
]

layernorm_fwd: object | None = None
layernorm_bwd_x: object | None = None
layernorm_gradw: object | None = None
layernorm_gradb: object | None = None


if not TYPE_CHECKING and triton_supported():
    try:  # pyright: ignore[reportUnreachable]
        import triton
        import triton.language as tl
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"Failed to import Triton, continuing: {e}")
    else:

        @triton.jit
        def layernorm_fwd(
            x_ptr,
            w_ptr,  # unused when HAS_WEIGHT=0
            b_ptr,  # unused when HAS_BIAS=0
            y_ptr,
            mean_ptr,
            inv_ptr,
            eps: tl.constexpr,
            D: tl.constexpr,
            stride_xr: tl.constexpr,
            stride_yr: tl.constexpr,
            HAS_WEIGHT: tl.constexpr,
            HAS_BIAS: tl.constexpr,
            USE_BF16: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            row = tl.program_id(0)
            cols = tl.arange(0, BLOCK)
            m = cols < D

            x = tl.load(x_ptr + row * stride_xr + cols, mask=m, other=0.0).to(tl.float32)
            mu = tl.sum(x, axis=0) / float(D)
            xc = x - mu
            var = tl.sum(xc * xc, axis=0) / float(D)
            inv = tl.rsqrt(var + float(eps))
            tl.store(mean_ptr + row, mu.to(tl.float32))
            tl.store(inv_ptr + row, inv.to(tl.float32))

            y = xc * inv
            if HAS_WEIGHT:
                w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
                y = y * w
            if HAS_BIAS:
                bb = tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
                y = y + bb

            if USE_BF16:
                tl.store(y_ptr + row * stride_yr + cols, y.to(tl.bfloat16), mask=m)
            else:
                tl.store(y_ptr + row * stride_yr + cols, y.to(tl.float16), mask=m)

        @triton.jit
        def layernorm_bwd_x(
            x_ptr,
            w_ptr,  # unused when HAS_WEIGHT=0
            mean_ptr,
            inv_ptr,
            gy_ptr,
            gx_ptr,
            D: tl.constexpr,
            stride_xr: tl.constexpr,
            stride_gyr: tl.constexpr,
            stride_gxr: tl.constexpr,
            HAS_WEIGHT: tl.constexpr,
            USE_BF16: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            row = tl.program_id(0)
            cols = tl.arange(0, BLOCK)
            m = cols < D

            x = tl.load(x_ptr + row * stride_xr + cols, mask=m, other=0.0).to(tl.float32)
            gy = tl.load(gy_ptr + row * stride_gyr + cols, mask=m, other=0.0).to(tl.float32)
            mu = tl.load(mean_ptr + row).to(tl.float32)
            inv = tl.load(inv_ptr + row).to(tl.float32)
            xc = x - mu

            g = gy
            if HAS_WEIGHT:
                w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
                g = gy * w

            sum_g = tl.sum(g, axis=0)
            sum_gx = tl.sum(g * xc, axis=0)
            rD = 1.0 / float(D)
            gx = inv * (g - sum_g * rD - xc * (inv * inv) * sum_gx * rD)

            if USE_BF16:
                tl.store(gx_ptr + row * stride_gxr + cols, gx.to(tl.bfloat16), mask=m)
            else:
                tl.store(gx_ptr + row * stride_gxr + cols, gx.to(tl.float16), mask=m)

        @triton.jit
        def layernorm_gradw(
            x_ptr,
            mean_ptr,
            inv_ptr,
            gy_ptr,
            gw_ptr,
            rows: tl.constexpr,
            D: tl.constexpr,
            stride_xr: tl.constexpr,
            stride_gyr: tl.constexpr,
            ROWS_PER_TILE: tl.constexpr,
            BLOCK_COL: tl.constexpr,
        ):
            pid_c = tl.program_id(0)
            pid_r = tl.program_id(1)
            cols = pid_c * BLOCK_COL + tl.arange(0, BLOCK_COL)
            m = cols < D

            r0 = pid_r * ROWS_PER_TILE
            acc = tl.zeros((BLOCK_COL,), dtype=tl.float32)
            for rr in range(ROWS_PER_TILE):
                r = r0 + rr
                rm = r < rows
                x = tl.load(x_ptr + r * stride_xr + cols, mask=m & rm, other=0.0).to(tl.float32)
                gy = tl.load(gy_ptr + r * stride_gyr + cols, mask=m & rm, other=0.0).to(tl.float32)
                mu = tl.load(mean_ptr + r, mask=rm, other=0.0).to(tl.float32)
                inv = tl.load(inv_ptr + r, mask=rm, other=0.0).to(tl.float32)
                acc += gy * (x - mu) * inv
            tl.atomic_add(gw_ptr + cols, acc, mask=m)

        @triton.jit
        def layernorm_gradb(
            gy_ptr,
            gb_ptr,
            rows: tl.constexpr,
            D: tl.constexpr,
            stride_gyr: tl.constexpr,
            ROWS_PER_TILE: tl.constexpr,
            BLOCK_COL: tl.constexpr,
        ):
            pid_c = tl.program_id(0)
            pid_r = tl.program_id(1)
            cols = pid_c * BLOCK_COL + tl.arange(0, BLOCK_COL)
            m = cols < D

            r0 = pid_r * ROWS_PER_TILE
            acc = tl.zeros((BLOCK_COL,), dtype=tl.float32)
            for rr in range(ROWS_PER_TILE):
                r = r0 + rr
                rm = r < rows
                gy = tl.load(gy_ptr + r * stride_gyr + cols, mask=m & rm, other=0.0).to(tl.float32)
                acc += gy
            tl.atomic_add(gb_ptr + cols, acc, mask=m)

