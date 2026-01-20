"""Triton CUDA backward kernel for SSM selective scan (blockwise)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from console import logger
from optimizer.runtime import triton_supported

__all__ = ["ssm_scan_block_bwd"]

ssm_scan_block_bwd: object | None = None


if not TYPE_CHECKING and triton_supported():
    try:  # pyright: ignore[reportUnreachable]
        import triton
        import triton.language as tl
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"Failed to import Triton, continuing: {e}")
    else:

        @triton.jit
        def ssm_scan_block_bwd(  # noqa: PLR0913
            x_ptr,
            dt_ptr,
            A_ptr,
            B_ptr,
            C_ptr,
            D_ptr,
            grad_y_ptr,
            h_out_ptr,
            ag_after_ptr,
            grad_x_ptr,
            grad_dt_ptr,
            grad_A_ptr,
            grad_B_ptr,
            grad_C_ptr,
            grad_D_ptr,
            ag_out_ptr,
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
            stride_gy_b: tl.constexpr,
            stride_gy_t: tl.constexpr,
            stride_gy_d: tl.constexpr,
            stride_gx_b: tl.constexpr,
            stride_gx_t: tl.constexpr,
            stride_gx_d: tl.constexpr,
            stride_gdt_b: tl.constexpr,
            stride_gdt_t: tl.constexpr,
            stride_gdt_d: tl.constexpr,
            stride_h_b: tl.constexpr,
            stride_h_d: tl.constexpr,
            stride_h_s: tl.constexpr,
            stride_gA_d: tl.constexpr,
            stride_gA_s: tl.constexpr,
            stride_gB_b: tl.constexpr,
            stride_gB_t: tl.constexpr,
            stride_gB_s: tl.constexpr,
            stride_gC_b: tl.constexpr,
            stride_gC_t: tl.constexpr,
            stride_gC_s: tl.constexpr,
            stride_gD: tl.constexpr,
            stride_ag_b: tl.constexpr,
            stride_ag_d: tl.constexpr,
            stride_ag_s: tl.constexpr,
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
            Dv = tl.load(D_ptr + d * stride_D, mask=m_d, other=0.0).to(tl.float32)

            h = tl.load(
                h_out_ptr + b * stride_h_b + d[:, None] * stride_h_d + s[None, :] * stride_h_s,
                mask=m_d[:, None] & m_s[None, :],
                other=0.0,
            ).to(tl.float32)
            ag_next = tl.load(
                ag_after_ptr + b * stride_ag_b + d[:, None] * stride_ag_d + s[None, :] * stride_ag_s,
                mask=m_d[:, None] & m_s[None, :],
                other=0.0,
            ).to(tl.float32)

            gradA_acc = tl.zeros((BLOCK_D, D_STATE), dtype=tl.float32)
            gradD_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

            for it in tl.static_range(BLOCK_T):
                t = t_start + (BLOCK_T - 1 - it)
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
                gy = tl.load(
                    grad_y_ptr + b * stride_gy_b + t * stride_gy_t + d * stride_gy_d,
                    mask=xm,
                    other=0.0,
                ).to(tl.float32)

                a = tl.exp(dts[:, None] * A)
                u = (dts * xs)[:, None] * Bs[None, :]
                h_prev = (h - u) / a

                g = gy[:, None] * Cs[None, :] + ag_next

                gx_u = tl.sum(g * (dts[:, None] * Bs[None, :]), axis=1)
                gx = gy * Dv + gx_u
                if USE_BF16:
                    tl.store(
                        grad_x_ptr + b * stride_gx_b + t * stride_gx_t + d * stride_gx_d,
                        gx.to(tl.bfloat16),
                        mask=xm,
                    )
                else:
                    tl.store(
                        grad_x_ptr + b * stride_gx_b + t * stride_gx_t + d * stride_gx_d,
                        gx.to(tl.float16),
                        mask=xm,
                    )

                gdt = tl.sum(g * (xs[:, None] * Bs[None, :] + h_prev * a * A), axis=1)
                if USE_BF16:
                    tl.store(
                        grad_dt_ptr + b * stride_gdt_b + t * stride_gdt_t + d * stride_gdt_d,
                        gdt.to(tl.bfloat16),
                        mask=xm,
                    )
                else:
                    tl.store(
                        grad_dt_ptr + b * stride_gdt_b + t * stride_gdt_t + d * stride_gdt_d,
                        gdt.to(tl.float16),
                        mask=xm,
                    )

                dt_x = dts * xs
                gB = tl.sum(g * dt_x[:, None], axis=0)
                gC = tl.sum(gy[:, None] * h, axis=0)
                tl.atomic_add(
                    grad_B_ptr + b * stride_gB_b + t * stride_gB_t + s * stride_gB_s,
                    gB,
                    mask=m_s & m_t,
                )
                tl.atomic_add(
                    grad_C_ptr + b * stride_gC_b + t * stride_gC_t + s * stride_gC_s,
                    gC,
                    mask=m_s & m_t,
                )

                gradA_acc += g * h_prev * a * dts[:, None]
                gradD_acc += gy * xs

                ag_next = a * g
                h = h_prev

            tl.atomic_add(
                grad_A_ptr + d[:, None] * stride_gA_d + s[None, :] * stride_gA_s,
                gradA_acc,
                mask=m_d[:, None] & m_s[None, :],
            )
            tl.atomic_add(grad_D_ptr + d * stride_gD, gradD_acc, mask=m_d)
            tl.store(
                ag_out_ptr + b * stride_ag_b + d[:, None] * stride_ag_d + s[None, :] * stride_ag_s,
                ag_next,
                mask=m_d[:, None] & m_s[None, :],
            )

