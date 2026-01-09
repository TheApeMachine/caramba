"""Triton CUDA kernels for RoPE (half-split layout)."""

from typing import TYPE_CHECKING

from caramba.console import logger
from caramba.optimizer.runtime import triton_supported

__all__ = ["rope_fwd", "rope_bwd"]

rope_fwd: object | None = None
rope_bwd: object | None = None


if not TYPE_CHECKING and triton_supported():
    try:  # pyright: ignore[reportUnreachable]
        import triton
        import triton.language as tl
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"Failed to import Triton, continuing: {e}")
    else:

        @triton.jit
        def rope_fwd(
            x_ptr,
            cos_ptr,
            sin_ptr,
            y_ptr,
            T: tl.constexpr,
            D: tl.constexpr,
            ROT: tl.constexpr,
            HALF: tl.constexpr,
            stride_xv: tl.constexpr,
            stride_xt: tl.constexpr,
            stride_yv: tl.constexpr,
            stride_yt: tl.constexpr,
            stride_cos_t: tl.constexpr,
            stride_cos_h: tl.constexpr,
            stride_sin_t: tl.constexpr,
            stride_sin_h: tl.constexpr,
            USE_BF16: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            vec = tl.program_id(0)
            blk = tl.program_id(1)
            cols = blk * BLOCK + tl.arange(0, BLOCK)
            m = cols < D
            t = vec % T

            x = tl.load(x_ptr + vec * stride_xv + cols * stride_xt, mask=m, other=0.0).to(tl.float32)
            # Rotate only where cols < HALF, and write both halves from those lanes.
            #
            # Note: Avoid `tl.any()` and tensor-dependent Python `if` blocks; older Triton
            # versions do not support them. Masked loads/stores already ensure correctness.
            m1 = m & (cols < HALF)
            c = tl.load(cos_ptr + t * stride_cos_t + cols * stride_cos_h, mask=m1, other=0.0).to(tl.float32)
            s = tl.load(sin_ptr + t * stride_sin_t + cols * stride_sin_h, mask=m1, other=0.0).to(tl.float32)
            x2 = tl.load(
                x_ptr + vec * stride_xv + (cols + HALF) * stride_xt,
                mask=m1 & ((cols + HALF) < D),
                other=0.0,
            ).to(tl.float32)
            y1 = x * c - x2 * s
            y2 = x * s + x2 * c

            if USE_BF16:
                tl.store(y_ptr + vec * stride_yv + cols * stride_yt, y1.to(tl.bfloat16), mask=m1)
                tl.store(
                    y_ptr + vec * stride_yv + (cols + HALF) * stride_yt,
                    y2.to(tl.bfloat16),
                    mask=m1 & ((cols + HALF) < D),
                )
            else:
                tl.store(y_ptr + vec * stride_yv + cols * stride_yt, y1.to(tl.float16), mask=m1)
                tl.store(
                    y_ptr + vec * stride_yv + (cols + HALF) * stride_yt,
                    y2.to(tl.float16),
                    mask=m1 & ((cols + HALF) < D),
                )

            # Passthrough for cols >= ROT.
            mp = m & (cols >= ROT)
            if USE_BF16:
                tl.store(y_ptr + vec * stride_yv + cols * stride_yt, x.to(tl.bfloat16), mask=mp)
            else:
                tl.store(y_ptr + vec * stride_yv + cols * stride_yt, x.to(tl.float16), mask=mp)

        @triton.jit
        def rope_bwd(
            gy_ptr,
            cos_ptr,
            sin_ptr,
            gx_ptr,
            T: tl.constexpr,
            D: tl.constexpr,
            ROT: tl.constexpr,
            HALF: tl.constexpr,
            stride_gyv: tl.constexpr,
            stride_gyt: tl.constexpr,
            stride_gxv: tl.constexpr,
            stride_gxt: tl.constexpr,
            stride_cos_t: tl.constexpr,
            stride_cos_h: tl.constexpr,
            stride_sin_t: tl.constexpr,
            stride_sin_h: tl.constexpr,
            USE_BF16: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            vec = tl.program_id(0)
            blk = tl.program_id(1)
            cols = blk * BLOCK + tl.arange(0, BLOCK)
            m = cols < D
            t = vec % T

            gy = tl.load(gy_ptr + vec * stride_gyv + cols * stride_gyt, mask=m, other=0.0).to(tl.float32)
            m1 = m & (cols < HALF)
            # Note: Avoid `tl.any()` and tensor-dependent Python `if` blocks; older Triton
            # versions do not support them. Masked loads/stores already ensure correctness.
            c = tl.load(cos_ptr + t * stride_cos_t + cols * stride_cos_h, mask=m1, other=0.0).to(tl.float32)
            s = tl.load(sin_ptr + t * stride_sin_t + cols * stride_sin_h, mask=m1, other=0.0).to(tl.float32)
            gy2 = tl.load(
                gy_ptr + vec * stride_gyv + (cols + HALF) * stride_gyt,
                mask=m1 & ((cols + HALF) < D),
                other=0.0,
            ).to(tl.float32)
            gx1 = gy * c + gy2 * s
            gx2 = -gy * s + gy2 * c

            if USE_BF16:
                tl.store(gx_ptr + vec * stride_gxv + cols * stride_gxt, gx1.to(tl.bfloat16), mask=m1)
                tl.store(
                    gx_ptr + vec * stride_gxv + (cols + HALF) * stride_gxt,
                    gx2.to(tl.bfloat16),
                    mask=m1 & ((cols + HALF) < D),
                )
            else:
                tl.store(gx_ptr + vec * stride_gxv + cols * stride_gxt, gx1.to(tl.float16), mask=m1)
                tl.store(
                    gx_ptr + vec * stride_gxv + (cols + HALF) * stride_gxt,
                    gx2.to(tl.float16),
                    mask=m1 & ((cols + HALF) < D),
                )

            mp = m & (cols >= ROT)
            if USE_BF16:
                tl.store(gx_ptr + vec * stride_gxv + cols * stride_gxt, gy.to(tl.bfloat16), mask=mp)
            else:
                tl.store(gx_ptr + vec * stride_gxv + cols * stride_gxt, gy.to(tl.float16), mask=mp)

