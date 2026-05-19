// SPDX-License-Identifier: Apache-2.0
// SSE2 stable-softmax helpers: exp-sum, scale, log-softmax shift.
#include "textflag.h"

DATA actSoftmaxClamp<>+0(SB)/4, $-87.0
GLOBL actSoftmaxClamp<>(SB), 8, $4

// func softmaxExpSumF32SSE2(dst, src *float32, maxValue float32, count int) float32
TEXT ·softmaxExpSumF32SSE2(SB), NOSPLIT, $0-20
	MOVL dst+0(FP), DI
	MOVL src+4(FP), SI
	MOVSS maxValue+8(FP), X6
	SHUFPS $0, X6, X6
	MOVL count+12(FP), CX
	MOVL $actX86ExpC<>(SB), AX
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
	MOVSS actX86Bias127<>(SB), X20
	PSHUFD $0, X20, X20
	MOVSS actSoftmaxClamp<>+0(SB), X8
	MOVAPS X8, X4
	SHUFPS $0, X4, X4
	XORPS X5, X5
smexp_sse2_w8:
	CMPL CX, $4
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
	MOVAPS X17, X7
	PADDD X1, X7
	ADDPS X7, X5
	MOVUPS X7, (DI)
	ADDL $16, SI
	ADDL $16, DI
	SUBL $4, CX
	JMP smexp_sse2_w8
smexp_sse2_w4:
	CMPL CX, $4
	JL smexp_sse2_scalar
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
	ADDL $16, SI
	ADDL $16, DI
	SUBL $4, CX
	JMP smexp_sse2_w4
smexp_sse2_scalar:
	TESTL CX, CX
	JZ smexp_sse2_fold
smexp_sse2_sloop:
	MOVSS (SI), X0
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
	MOVSS X7, (DI)
	ADDL $4, SI
	ADDL $4, DI
	DECL CX
	JNZ smexp_sse2_sloop
smexp_sse2_fold:
	MOVAPS X5, X0
	HADDPS X5, X0
	HADDPS X0, X0
	HADDPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func scaleF32SSE2(dst, src *float32, scale float32, count int)
TEXT ·scaleF32SSE2(SB), NOSPLIT, $0-16
	MOVL dst+0(FP), DI
	MOVL src+4(FP), SI
	MOVSS scale+8(FP), X8
	SHUFPS $0, X8, X8
	MOVL count+12(FP), CX
smscale_sse2_w8:
	CMPL CX, $4
	JL smscale_sse2_w4
	MOVUPS (SI), X0
	MULPS X8, X0
	MOVUPS X0, (DI)
	ADDL $16, SI
	ADDL $16, DI
	SUBL $4, CX
	JMP smscale_sse2_w8
smscale_sse2_w4:
	CMPL CX, $4
	JL smscale_sse2_scalar
	MOVUPS (SI), X0
	MULPS X8, X0
	MOVUPS X0, (DI)
	ADDL $16, SI
	ADDL $16, DI
	SUBL $4, CX
	JMP smscale_sse2_w4
smscale_sse2_scalar:
	TESTL CX, CX
	JZ smscale_sse2_done
smscale_sse2_sloop:
	MOVSS (SI), X0
	MULPS X8, X0
	MOVSS X0, (DI)
	ADDL $4, SI
	ADDL $4, DI
	DECL CX
	JNZ smscale_sse2_sloop
smscale_sse2_done:
	RET

// func logSoftmaxShiftF32SSE2(dst, src *float32, maxValue, logSum float32, count int)
TEXT ·logSoftmaxShiftF32SSE2(SB), NOSPLIT, $0-20
	MOVL dst+0(FP), DI
	MOVL src+4(FP), SI
	MOVSS maxValue+8(FP), X8
	SHUFPS $0, X8, X8
	MOVSS logSum+12(FP), X9
	SHUFPS $0, X9, X9
	MOVL count+16(FP), CX
smlog_sse2_w8:
	CMPL CX, $4
	JL smlog_sse2_w4
	MOVUPS (SI), X0
	SUBPS X8, X0
	MOVAPS X0, X7
	SUBPS X9, X7
	MOVUPS X7, (DI)
	ADDL $16, SI
	ADDL $16, DI
	SUBL $4, CX
	JMP smlog_sse2_w8
smlog_sse2_w4:
	CMPL CX, $4
	JL smlog_sse2_scalar
	MOVUPS (SI), X0
	SUBPS X8, X0
	MOVAPS X0, X7
	SUBPS X9, X7
	MOVUPS X7, (DI)
	ADDL $16, SI
	ADDL $16, DI
	SUBL $4, CX
	JMP smlog_sse2_w4
smlog_sse2_scalar:
	TESTL CX, CX
	JZ smlog_sse2_done
smlog_sse2_sloop:
	MOVSS (SI), X0
	SUBPS X8, X0
	MOVAPS X0, X7
	SUBPS X9, X7
	MOVSS X7, (DI)
	ADDL $4, SI
	ADDL $4, DI
	DECL CX
	JNZ smlog_sse2_sloop
smlog_sse2_done:
	RET

// func reduceMaxSoftmaxF32SSE2(src *float32, count int) float32
TEXT ·reduceMaxSoftmaxF32SSE2(SB), NOSPLIT, $0-8
	MOVL src+0(FP), SI
	MOVL count+4(FP), CX
	TESTL CX, CX
	JZ reduce_max_sse2_zero
	MOVSS (SI), X0
	SHUFPS $0, X0, X0
	ADDL $4, SI
	DECL CX
reduce_max_sse2_w4:
	CMPL CX, $4
	JL reduce_max_sse2_extract
	MOVUPS (SI), X1
	MAXPS X1, X0
	ADDL $16, SI
	SUBL $4, CX
	JMP reduce_max_sse2_w4
reduce_max_sse2_extract:
	MOVAPS X0, X1
	HADDPS X1, X0
	HADDPS X0, X0
reduce_max_sse2_tail:
	TESTL CX, CX
	JZ reduce_max_sse2_done
reduce_max_sse2_sloop:
	MAXSS (SI), X0
	ADDL $4, SI
	DECL CX
	JNZ reduce_max_sse2_sloop
	JMP reduce_max_sse2_done
reduce_max_sse2_done:
	MOVSS X0, ret+8(FP)
	RET
reduce_max_sse2_zero:
	XORPS X0, X0
	MOVSS X0, ret+8(FP)
	RET

// func SoftmaxF32SSE2(dst, src *float32, count int)
TEXT ·SoftmaxF32SSE2(SB), NOSPLIT, $32-12
	MOVL dst+0(FP), BX
	MOVL src+4(FP), SI
	MOVL count+8(FP), CX
	MOVL SI, 0(SP)
	MOVL CX, 4(SP)
	CALL ·reduceMaxSoftmaxF32SSE2(SB)
	MOVSS X0, X6
	MOVL BX, 0(SP)
	MOVL SI, 4(SP)
	MOVSS X6, 8(SP)
	MOVL CX, 12(SP)
	CALL ·softmaxExpSumF32SSE2(SB)
	MOVSS actX86LogC<>+4(SB), X1
	DIVSS X0, X1
	MOVL BX, 0(SP)
	MOVL BX, 4(SP)
	MOVSS X1, 8(SP)
	MOVL CX, 12(SP)
	CALL ·scaleF32SSE2(SB)
	RET

// func LogSoftmaxF32SSE2(dst, src *float32, count int)
TEXT ·LogSoftmaxF32SSE2(SB), NOSPLIT, $32-12
	MOVL dst+0(FP), BX
	MOVL src+4(FP), SI
	MOVL count+8(FP), CX
	MOVL SI, 0(SP)
	MOVL CX, 4(SP)
	CALL ·reduceMaxSoftmaxF32SSE2(SB)
	MOVSS X0, X6
	MOVL BX, 0(SP)
	MOVL SI, 4(SP)
	MOVSS X6, 8(SP)
	MOVL CX, 12(SP)
	CALL ·softmaxExpSumF32SSE2(SB)
	MOVSS X0, 0(SP)
	LEAL 0(SP), AX
	MOVL AX, 0(SP)
	MOVL AX, 4(SP)
	MOVL $1, 8(SP)
	CALL ·LogF32SSE2(SB)
	MOVSS X0, X7
	MOVL BX, 0(SP)
	MOVL SI, 4(SP)
	MOVSS X6, 8(SP)
	MOVSS X7, 12(SP)
	MOVL CX, 16(SP)
	CALL ·logSoftmaxShiftF32SSE2(SB)
	RET
