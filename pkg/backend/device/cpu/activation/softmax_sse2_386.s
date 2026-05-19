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
	VBROADCASTSS X6, X6
	MOVL count+12(FP), CX
	MOVL $actX86ExpC<>(SB), AX
	VMOVSS (AX), X8
	VBROADCASTSS X8, X8
	VMOVSS 4(AX), X9
	VBROADCASTSS X9, X9
	VMOVSS 12(AX), X11
	VBROADCASTSS X11, X11
	VMOVSS 16(AX), X12
	VBROADCASTSS X12, X12
	VMOVSS 20(AX), X13
	VBROADCASTSS X13, X13
	VMOVSS 24(AX), X14
	VBROADCASTSS X14, X14
	VMOVSS 28(AX), X15
	VBROADCASTSS X15, X15
	VMOVSS 32(AX), X16
	VBROADCASTSS X16, X16
	VMOVSS 36(AX), X17
	VBROADCASTSS X17, X17
	VPBROADCASTD actX86Bias127<>(SB), X20
	VMOVSS actSoftmaxClamp<>+0(SB), X8
	VBROADCASTSS X8, X4
	VXORPS X5, X5, X5
smexp_sse2_w8:
	CMPL CX, $4
	JL smexp_sse2_w4
	VMOVUPS (SI), X0
	VSUBPS X6, X0, X0
	VMAXPS X4, X0, X0
	VMULPS X8, X0, X1
	VROUNDPS $8, X1, X1
	VMULPS X1, X9, X2
	VSUBPS X2, X0, X0
	VMOVAPS X11, X3
	VFMADD213PS X3, X0, X11
	VMOVAPS X12, X3
	VFMADD213PS X3, X0, X12
	VMOVAPS X13, X3
	VFMADD213PS X3, X0, X13
	VMOVAPS X14, X3
	VFMADD213PS X3, X0, X14
	VMOVAPS X15, X3
	VFMADD213PS X3, X0, X15
	VMOVAPS X16, X3
	VFMADD213PS X3, X0, X16
	VMOVAPS X17, X7
	VFMADD213PS X7, X0, X17
	VCVTPS2DQ X1, X1
	VPADDD X20, X1, X1
	VPSLLD $23, X1, X1
	VPADDD X1, X7, X7
	VADDPS X5, X7, X5
	VMOVUPS X7, (DI)
	ADDL $16, SI
	ADDL $16, DI
	SUBL $4, CX
	JMP smexp_sse2_w8
smexp_sse2_w4:
	CMPL CX, $4
	JL smexp_sse2_scalar
	VMOVUPS (SI), X0
	VSUBPS X6, X0, X0
	VMAXPS X4, X0, X0
	VMULPS X8, X0, X1
	VROUNDPS $8, X1, X1
	VMULPS X1, X9, X2
	VSUBPS X2, X0, X0
	VMOVAPS X11, X3
	VFMADD213PS X3, X0, X11
	VMOVAPS X12, X3
	VFMADD213PS X3, X0, X12
	VMOVAPS X13, X3
	VFMADD213PS X3, X0, X13
	VMOVAPS X14, X3
	VFMADD213PS X3, X0, X14
	VMOVAPS X15, X3
	VFMADD213PS X3, X0, X15
	VMOVAPS X16, X3
	VFMADD213PS X3, X0, X16
	VMOVAPS X17, X7
	VFMADD213PS X7, X0, X17
	VCVTPS2DQ X1, X1
	VPADDD X20, X1, X1
	VPSLLD $23, X1, X1
	VPADDD X1, X7, X7
	VADDPS X5, X7, X5
	VMOVUPS X7, (DI)
	ADDL $16, SI
	ADDL $16, DI
	SUBL $4, CX
	JMP smexp_sse2_w4
smexp_sse2_scalar:
	TESTL CX, CX
	JZ smexp_sse2_fold
smexp_sse2_sloop:
	MOVSS (SI), X0
	VSUBPS X6, X0, X0
	VMAXPS X4, X0, X0
	VMULPS X8, X0, X1
	VROUNDPS $8, X1, X1
	VMULPS X1, X9, X2
	VSUBPS X2, X0, X0
	VMOVAPS X11, X3
	VFMADD213PS X3, X0, X11
	VMOVAPS X12, X3
	VFMADD213PS X3, X0, X12
	VMOVAPS X13, X3
	VFMADD213PS X3, X0, X13
	VMOVAPS X14, X3
	VFMADD213PS X3, X0, X14
	VMOVAPS X15, X3
	VFMADD213PS X3, X0, X15
	VMOVAPS X16, X3
	VFMADD213PS X3, X0, X16
	VMOVAPS X17, X7
	VFMADD213PS X7, X0, X17
	VCVTPS2DQ X1, X1
	VPADDD X20, X1, X1
	VPSLLD $23, X1, X1
	VPADDD X1, X7, X7
	VADDPS X5, X7, X5
	MOVSS X7, (DI)
	ADDL $4, SI
	ADDL $4, DI
	DECL CX
	JNZ smexp_sse2_sloop
smexp_sse2_fold:
	VHADDPS X5, X5, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func scaleF32SSE2(dst, src *float32, scale float32, count int)
TEXT ·scaleF32SSE2(SB), NOSPLIT, $0-16
	MOVL dst+0(FP), DI
	MOVL src+4(FP), SI
	VMOVSS scale+8(FP), X8
	VBROADCASTSS X8, X8
	MOVL count+12(FP), CX
smscale_sse2_w8:
	CMPL CX, $4
	JL smscale_sse2_w4
	VMOVUPS (SI), X0
	VMULPS X8, X0, X0
	VMOVUPS X0, (DI)
	ADDL $16, SI
	ADDL $16, DI
	SUBL $4, CX
	JMP smscale_sse2_w8
smscale_sse2_w4:
	CMPL CX, $4
	JL smscale_sse2_scalar
	VMOVUPS (SI), X0
	VMULPS X8, X0, X0
	VMOVUPS X0, (DI)
	ADDL $16, SI
	ADDL $16, DI
	SUBL $4, CX
	JMP smscale_sse2_w4
smscale_sse2_scalar:
	TESTL CX, CX
	JZ smscale_sse2_done
smscale_sse2_sloop:
	MOVSS (SI), X0
	VMULPS X8, X0, X0
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
	VMOVSS maxValue+8(FP), X8
	VBROADCASTSS X8, X8
	VMOVSS logSum+12(FP), X9
	VBROADCASTSS X9, X9
	MOVL count+16(FP), CX
smlog_sse2_w8:
	CMPL CX, $4
	JL smlog_sse2_w4
	VMOVUPS (SI), X0
	VSUBPS X8, X0, X0
	VSUBPS X9, X0, X7
	VMOVUPS X7, (DI)
	ADDL $16, SI
	ADDL $16, DI
	SUBL $4, CX
	JMP smlog_sse2_w8
smlog_sse2_w4:
	CMPL CX, $4
	JL smlog_sse2_scalar
	VMOVUPS (SI), X0
	VSUBPS X8, X0, X0
	VSUBPS X9, X0, X7
	VMOVUPS X7, (DI)
	ADDL $16, SI
	ADDL $16, DI
	SUBL $4, CX
	JMP smlog_sse2_w4
smlog_sse2_scalar:
	TESTL CX, CX
	JZ smlog_sse2_done
smlog_sse2_sloop:
	MOVSS (SI), X0
	VSUBPS X8, X0, X0
	VSUBPS X9, X0, X7
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
	VBROADCASTSS X0, X0
	ADDL $4, SI
	DECL CX
reduce_max_sse2_w4:
	CMPL CX, $4
	JL reduce_max_sse2_extract
	VMOVUPS (SI), X1
	VMAXPS X1, X0, X0
	ADDL $16, SI
	SUBL $4, CX
	JMP reduce_max_sse2_w4
reduce_max_sse2_extract:
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
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
	VMOVSS actX86LogC<>+4(SB), X1
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
