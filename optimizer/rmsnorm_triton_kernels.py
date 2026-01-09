"""Triton CUDA kernels for RMSNorm."""

from __future__ import annotations

from typing import TYPE_CHECKING

from caramba.console import logger
from caramba.optimizer.runtime import triton_supported

__all__ = ["rmsnorm_fwd", "rmsnorm_bwd_x", "rmsnorm_bwd_x_noweight", "rmsnorm_bwd_w"]

rmsnorm_fwd: object | None = None
rmsnorm_bwd_x: object | None = None
rmsnorm_bwd_x_noweight: object | None = None
rmsnorm_bwd_w: object | None = None


if not TYPE_CHECKING and triton_supported():
    try:  # pyright: ignore[reportUnreachable]
        import triton
        import triton.language as tl
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"Failed to import Triton, continuing: {e}")
    else:

        @triton.jit
        def rmsnorm_fwd(
            x_ptr,
            w_ptr,  # can be unused when HAS_WEIGHT=0
            y_ptr,
            inv_ptr,
            eps: tl.constexpr,
            D: tl.constexpr,
            stride_xr: tl.constexpr,
            stride_yr: tl.constexpr,
            HAS_WEIGHT: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            row = tl.program_id(0)
            cols = tl.arange(0, BLOCK)
            m = cols < D

            x = tl.load(x_ptr + row * stride_xr + cols, mask=m, other=0.0).to(tl.float32)
            ms = tl.sum(x * x, axis=0) / float(D)
            inv = tl.rsqrt(ms + float(eps))
            tl.store(inv_ptr + row, inv.to(tl.float32))

            if HAS_WEIGHT:
                w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
                y = x * inv * w
            else:
                y = x * inv
            tl.store(y_ptr + row * stride_yr + cols, y.to(tl.float32), mask=m)

        @triton.jit
        def rmsnorm_bwd_x(
            x_ptr,
            w_ptr,
            inv_ptr,
            gy_ptr,
            gx_ptr,
            D: tl.constexpr,
            stride_xr: tl.constexpr,
            stride_gyr: tl.constexpr,
            stride_gxr: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            row = tl.program_id(0)
            cols = tl.arange(0, BLOCK)
            m = cols < D

            x = tl.load(x_ptr + row * stride_xr + cols, mask=m, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
            gy = tl.load(gy_ptr + row * stride_gyr + cols, mask=m, other=0.0).to(tl.float32)
            inv = tl.load(inv_ptr + row).to(tl.float32)

            g = gy * w
            dot = tl.sum(g * x, axis=0)
            coeff = (inv * inv * inv) * (dot / float(D))
            gx = g * inv - x * coeff
            tl.store(gx_ptr + row * stride_gxr + cols, gx.to(tl.float32), mask=m)

        @triton.jit
        def rmsnorm_bwd_x_noweight(
            x_ptr,
            inv_ptr,
            gy_ptr,
            gx_ptr,
            D: tl.constexpr,
            stride_xr: tl.constexpr,
            stride_gyr: tl.constexpr,
            stride_gxr: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            row = tl.program_id(0)
            cols = tl.arange(0, BLOCK)
            m = cols < D

            x = tl.load(x_ptr + row * stride_xr + cols, mask=m, other=0.0).to(tl.float32)
            gy = tl.load(gy_ptr + row * stride_gyr + cols, mask=m, other=0.0).to(tl.float32)
            inv = tl.load(inv_ptr + row).to(tl.float32)

            dot = tl.sum(gy * x, axis=0)
            coeff = (inv * inv * inv) * (dot / float(D))
            gx = gy * inv - x * coeff
            tl.store(gx_ptr + row * stride_gxr + cols, gx.to(tl.float32), mask=m)

        @triton.jit
        def rmsnorm_bwd_w(
            x_ptr,
            inv_ptr,
            gy_ptr,
            gw_ptr,
            rows,
            D: tl.constexpr,
            stride_xr: tl.constexpr,
            stride_gyr: tl.constexpr,
            ROWS_PER_TILE: tl.constexpr,
            BLOCK_COL: tl.constexpr,
        ):
            """Compute grad_w with a tiled reduction over rows and atomic adds.

            Note: A full `for r in range(rows)` unroll is not viable; we instead
            parallelize across row tiles and accumulate into `gw_ptr`.
            """
            pid_c = tl.program_id(0)
            pid_r = tl.program_id(1)
            cols = pid_c * BLOCK_COL + tl.arange(0, BLOCK_COL)
            m = cols < D

            r0 = pid_r * ROWS_PER_TILE
            acc = tl.zeros((BLOCK_COL,), dtype=tl.float32)
            for rr in range(ROWS_PER_TILE):
                r = r0 + rr
                rm = r < rows
                x = tl.load(x_ptr + r * stride_xr + cols, mask=rm & m, other=0.0).to(tl.float32)
                gy = tl.load(gy_ptr + r * stride_gyr + cols, mask=rm & m, other=0.0).to(tl.float32)
                inv = tl.load(inv_ptr + r, mask=rm, other=0.0).to(tl.float32)
                acc += gy * x * inv
            tl.atomic_add(gw_ptr + cols, acc, mask=m)
