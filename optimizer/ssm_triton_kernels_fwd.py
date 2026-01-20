"""Triton CUDA forward kernel for SSM selective scan (blockwise)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from console import logger
from optimizer.runtime import triton_supported

__all__ = ["ssm_scan_block_fwd"]

ssm_scan_block_fwd: object | None = None


if not TYPE_CHECKING and triton_supported():
    try:  # pyright: ignore[reportUnreachable]
        import triton
        import triton.language as tl
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"Failed to import Triton, continuing: {e}")
    else:

        @triton.jit
        def ssm_scan_block_fwd(  # noqa: PLR0913
            x_ptr,
            dt_ptr,
            A_ptr,
            B_ptr,
            C_ptr,
            D_ptr,
            y_ptr,
            h_in_ptr,
            h_out_ptr,
            t_start: tl.int32,
            T: tl.int32,
            D_inner: tl.int32,
            stride_x_b: tl.constexpr,
            stride_x_t: tl.constexpr,
            stride_x_d: tl.constexpr,
            stride_dt_b: tl.constexpr,
            stride_dt_t: tl.constexpr,
            stride_dt_d: tl.constexpr,
            stride_A_d: tl.constexpr,
            stride_A_s: tl.constexpr,
            stride_B_b: tl.constexpr,
            stride_B_t: tl.constexpr,
            stride_B_s: tl.constexpr,
            stride_C_b: tl.constexpr,
            stride_C_t: tl.constexpr,
            stride_C_s: tl.constexpr,
            stride_D: tl.constexpr,
            stride_y_b: tl.constexpr,
            stride_y_t: tl.constexpr,
            stride_y_d: tl.constexpr,
            stride_h_b: tl.constexpr,
            stride_h_d: tl.constexpr,
            stride_h_s: tl.constexpr,
            D_STATE: tl.constexpr,
            BLOCK_T: tl.constexpr,
            BLOCK_D: tl.constexpr,
            USE_BF16: tl.constexpr,
        ):
            pid = tl.program_id(0)
            num_d_blocks = tl.cdiv(D_inner, BLOCK_D)
            b = pid // num_d_blocks
            d_block = pid - b * num_d_blocks

            d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
            m_d = d < D_inner

            s = tl.arange(0, D_STATE)
            m_s = s < D_STATE

            A = tl.load(
                A_ptr + d[:, None] * stride_A_d + s[None, :] * stride_A_s,
                mask=m_d[:, None] & m_s[None, :],
                other=0.0,
            ).to(tl.float32)
            h = tl.load(
                h_in_ptr + b * stride_h_b + d[:, None] * stride_h_d + s[None, :] * stride_h_s,
                mask=m_d[:, None] & m_s[None, :],
                other=0.0,
            ).to(tl.float32)
            Dv = tl.load(D_ptr + d * stride_D, mask=m_d, other=0.0).to(tl.float32)

            for it in tl.static_range(BLOCK_T):
                t = t_start + it
                m_t = t < T
                xm = m_d & m_t

                xs = tl.load(
                    x_ptr + b * stride_x_b + t * stride_x_t + d * stride_x_d,
                    mask=xm,
                    other=0.0,
                ).to(tl.float32)
                dts = tl.load(
                    dt_ptr + b * stride_dt_b + t * stride_dt_t + d * stride_dt_d,
                    mask=xm,
                    other=0.0,
                ).to(tl.float32)
                Bs = tl.load(
                    B_ptr + b * stride_B_b + t * stride_B_t + s * stride_B_s,
                    mask=m_s & m_t,
                    other=0.0,
                ).to(tl.float32)
                Cs = tl.load(
                    C_ptr + b * stride_C_b + t * stride_C_t + s * stride_C_s,
                    mask=m_s & m_t,
                    other=0.0,
                ).to(tl.float32)

                a = tl.exp(dts[:, None] * A)
                u = (dts * xs)[:, None] * Bs[None, :]
                h = a * h + u
                y = tl.sum(h * Cs[None, :], axis=1) + Dv * xs

                if USE_BF16:
                    tl.store(
                        y_ptr + b * stride_y_b + t * stride_y_t + d * stride_y_d,
                        y.to(tl.bfloat16),
                        mask=xm,
                    )
                else:
                    tl.store(
                        y_ptr + b * stride_y_b + t * stride_y_t + d * stride_y_d,
                        y.to(tl.float16),
                        mask=xm,
                    )

            tl.store(
                h_out_ptr + b * stride_h_b + d[:, None] * stride_h_d + s[None, :] * stride_h_s,
                h,
                mask=m_d[:, None] & m_s[None, :],
            )

