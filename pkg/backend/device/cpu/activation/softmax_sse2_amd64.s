// SPDX-License-Identifier: Apache-2.0
// SSE2 stable-softmax helpers: exp-sum, scale, log-softmax shift.
#include "textflag.h"

DATA actSoftmaxClamp<>+0(SB)/4, $-87.0
GLOBL actSoftmaxClamp<>(SB), 8, $4

// func softmaxExpSumF32SSE2(dst, src *float32, maxValue float32, count int) float32
TEXT ·softmaxExpSumF32SSE2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVSS maxValue+16(FP), X6
	SHUFPS $0, X6, X6
	MOVQ count+24(FP), CX
	MOVQ $actX86ExpC<>(SB), AX
	MOVSS (AX), X8
	SHUFPS $0, X8, X8
	MOVSS 4(AX), X9
	SHUFPS $0, X9, X9
	MOVSS 12(AX), X11
	SHUFPS $0, X11, X11
	MOVSS 16(AX), X12
	SHUFPS $0, X12, X12
	MOVSS 20(AX), X13
	SHUFPS $0, X13, X13
	MOVSS 24(AX), X14
	SHUFPS $0, X14, X14
	MOVSS 28(AX), X15
	SHUFPS $0, X15, X15
	MOVSS 32(AX), X16
	SHUFPS $0, X16, X16
	MOVSS 36(AX), X17
	SHUFPS $0, X17, X17
	// VPBROADCASTD -> MOVD + PSHUFD
	MOVSS actX86Bias127<>(SB), X20
	PSHUFD $0, X20, X20
	MOVSS actSoftmaxClamp<>+0(SB), X8
	MOVAPS X8, X4
	SHUFPS $0, X4, X4
	XORPS X5, X5
smexp_sse2_w8:
	CMPQ CX, $4
	JL smexp_sse2_w4
	MOVUPS (SI), X0
	SUBPS X6, X0
	MAXPS X4, X0
	MOVAPS X0, X1
	MULPS X8, X1
	ROUNDPS $8, X1, X1
	MOVAPS X1, X2
	MULPS X9, X2
	SUBPS X2, X0
	MOVAPS X11, X3
	MULPS X0, X11
	ADDPS X3, X11
	MOVAPS X12, X3
	MULPS X0, X12
	ADDPS X3, X12
	MOVAPS X13, X3
	MULPS X0, X13
	ADDPS X3, X13
	MOVAPS X14, X3
	MULPS X0, X14
	ADDPS X3, X14
	MOVAPS X15, X3
	MULPS X0, X15
	ADDPS X3, X15
	MOVAPS X16, X3
	MULPS X0, X16
	ADDPS X3, X16
	MOVAPS X17, X7
	MULPS X0, X17
	ADDPS X7, X17
	CVTPS2DQ X1, X1
	PADDD X20, X1
	PSLLD $23, X1
	PADDD X1, X7
	ADDPS X7, X5
	MOVUPS X7, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP smexp_sse2_w8
smexp_sse2_w4:
	CMPQ CX, $4
	JL smexp_sse2_done
	MOVUPS (SI), X0
	SUBPS X6, X0
	MAXPS X4, X0
	MOVAPS X0, X1
	MULPS X8, X1
	ROUNDPS $8, X1, X1
	MOVAPS X1, X2
	MULPS X9, X2
	SUBPS X2, X0
	MOVAPS X11, X3
	MULPS X0, X11
	ADDPS X3, X11
	MOVAPS X12, X3
	MULPS X0, X12
	ADDPS X3, X12
	MOVAPS X13, X3
	MULPS X0, X13
	ADDPS X3, X13
	MOVAPS X14, X3
	MULPS X0, X14
	ADDPS X3, X14
	MOVAPS X15, X3
	MULPS X0, X15
	ADDPS X3, X15
	MOVAPS X16, X3
	MULPS X0, X16
	ADDPS X3, X16
	MOVAPS X17, X7
	MULPS X0, X17
	ADDPS X7, X17
	CVTPS2DQ X1, X1
	PADDD X20, X1
	PSLLD $23, X1
	MOVAPS X17, X7
	PADDD X1, X7
	ADDPS X7, X5
	MOVUPS X7, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP smexp_sse2_w4
smscale_sse2_done:
	RET

// func logSoftmaxShiftF32SSE2(dst, src *float32, maxValue, logSum float32, count int)
TEXT ·logSoftmaxShiftF32SSE2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVSS maxValue+16(FP), X8
	SHUFPS $0, X8, X8
	MOVSS logSum+20(FP), X9
	SHUFPS $0, X9, X9
	MOVQ count+24(FP), CX
smlog_sse2_w8:
	CMPQ CX, $4
	JL smlog_sse2_w4
	MOVUPS (SI), X0
	SUBPS X8, X0
	MOVAPS X0, X7
	SUBPS X9, X7
	MOVUPS X7, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP smlog_sse2_w8
smlog_sse2_w4:
	CMPQ CX, $4
	JL smlog_sse2_done
	MOVUPS (SI), X0
	SUBPS X8, X0
	MOVAPS X0, X7
	SUBPS X9, X7
	MOVUPS X7, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP smlog_sse2_w4
smlog_sse2_done:
	RET

// func reduceMaxSoftmaxF32SSE2(src *float32, count int) float32
TEXT ·reduceMaxSoftmaxF32SSE2(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ reduce_max_sse2_zero
	MOVSS (SI), X0
	SHUFPS $0, X0, X0
	ADDQ $4, SI
	DECQ CX
reduce_max_sse2_w4:
	CMPQ CX, $4
	JL reduce_max_sse2_extract
	MOVUPS (SI), X1
	MAXPS X1, X0
	ADDQ $16, SI
	SUBQ $4, CX
	JMP reduce_max_sse2_w4
reduce_max_sse2_extract:
	MOVAPS X0, X1
	HADDPS X1, X0
	HADDPS X0, X0
reduce_max_sse2_tail:
	TESTQ CX, CX
	JZ reduce_max_sse2_done
reduce_max_sse2_done:
	MOVSS X0, ret+16(FP)
	RET
reduce_max_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func SoftmaxF32SSE2(dst, src *float32, count int)
TEXT ·SoftmaxF32SSE2(SB), NOSPLIT, $32-24
	MOVQ dst+0(FP), R10
	MOVQ src+8(FP), R11
	MOVQ count+16(FP), R12
	MOVQ R11, 0(SP)
	MOVQ R12, 8(SP)
	CALL ·reduceMaxSoftmaxF32SSE2(SB)
	MOVSS X0, X6
	MOVQ R10, 0(SP)
	MOVQ R11, 8(SP)
	MOVSS X6, 16(SP)
	MOVQ R12, 24(SP)
	CALL ·softmaxExpSumF32SSE2(SB)
	MOVSS actX86LogC<>+4(SB), X8
	DIVSS X0, X8
	MOVQ R10, 0(SP)
	MOVQ R10, 8(SP)
	MOVSS X8, 16(SP)
	MOVQ R12, 24(SP)
	CALL ·scaleF32SSE2(SB)
	RET

// func LogSoftmaxF32SSE2(dst, src *float32, count int)
TEXT ·LogSoftmaxF32SSE2(SB), NOSPLIT, $32-24
	MOVQ dst+0(FP), R10
	MOVQ src+8(FP), R11
	MOVQ count+16(FP), R12
	MOVQ R11, 0(SP)
	MOVQ R12, 8(SP)
	CALL ·reduceMaxSoftmaxF32SSE2(SB)
	MOVSS X0, X6
	MOVQ R10, 0(SP)
	MOVQ R11, 8(SP)
	MOVSS X6, 16(SP)
	MOVQ R12, 24(SP)
	CALL ·softmaxExpSumF32SSE2(SB)
	MOVSS X0, 0(SP)
	LEAQ 0(SP), AX
	MOVQ AX, 0(SP)
	MOVQ AX, 8(SP)
	MOVQ $1, 16(SP)
	CALL ·LogF32SSE2(SB)
	MOVSS X0, X9
	MOVQ R10, 0(SP)
	MOVQ R11, 8(SP)
	MOVSS X6, 16(SP)
	MOVSS X9, 20(SP)
	MOVQ R12, 24(SP)
	CALL ·logSoftmaxShiftF32SSE2(SB)
	RET
