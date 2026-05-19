// SPDX-License-Identifier: Apache-2.0
// AVX2 stable-softmax helpers: exp-sum, scale, log-softmax shift.
#include "textflag.h"

DATA actSoftmaxClamp<>+0(SB)/4, $-87.0
GLOBL actSoftmaxClamp<>(SB), 8, $4

// func softmaxExpSumF32AVX2(dst, src *float32, maxValue float32, count int) float32
TEXT ·softmaxExpSumF32AVX2(SB), NOSPLIT, $0-20
	MOVL dst+0(FP), DI
	MOVL src+4(FP), SI
	MOVSS maxValue+8(FP), X6
	VBROADCASTSS X6, Y6
	MOVL count+12(FP), CX
	MOVL $actX86ExpC<>(SB), AX
	VMOVSS (AX), X8
	VBROADCASTSS X8, Y8
	VMOVSS 4(AX), X9
	VBROADCASTSS X9, Y9
	VMOVSS 12(AX), X11
	VBROADCASTSS X11, Y11
	VMOVSS 16(AX), X12
	VBROADCASTSS X12, Y12
	VMOVSS 20(AX), X13
	VBROADCASTSS X13, Y13
	VMOVSS 24(AX), X14
	VBROADCASTSS X14, Y14
	VMOVSS 28(AX), X15
	VBROADCASTSS X15, Y15
	VMOVSS 32(AX), X16
	VBROADCASTSS X16, Y16
	VMOVSS 36(AX), X17
	VBROADCASTSS X17, Y17
	VPBROADCASTD actX86Bias127<>(SB), Y20
	VMOVSS actSoftmaxClamp<>+0(SB), X8
	VBROADCASTSS X8, Y4
	VXORPS Y5, Y5, Y5
smexp_avx2_w8:
	CMPL CX, $8
	JL smexp_avx2_w4
	VMOVUPS (SI), Y0
	VSUBPS Y6, Y0, Y0
	VMAXPS Y4, Y0, Y0
	VMULPS Y8, Y0, Y1
	VROUNDPS $8, Y1, Y1
	VMULPS Y1, Y9, Y2
	VSUBPS Y2, Y0, Y0
	VMOVAPS Y11, Y3
	VFMADD213PS Y3, Y0, Y11
	VMOVAPS Y12, Y3
	VFMADD213PS Y3, Y0, Y12
	VMOVAPS Y13, Y3
	VFMADD213PS Y3, Y0, Y13
	VMOVAPS Y14, Y3
	VFMADD213PS Y3, Y0, Y14
	VMOVAPS Y15, Y3
	VFMADD213PS Y3, Y0, Y15
	VMOVAPS Y16, Y3
	VFMADD213PS Y3, Y0, Y16
	VMOVAPS Y17, Y7
	VFMADD213PS Y7, Y0, Y17
	VCVTPS2DQ Y1, Y1
	VPADDD Y20, Y1, Y1
	VPSLLD $23, Y1, Y1
	VPADDD Y1, Y7, Y7
	VADDPS Y5, Y7, Y5
	VMOVUPS Y7, (DI)
	ADDL $32, SI
	ADDL $32, DI
	SUBL $8, CX
	JMP smexp_avx2_w8
smexp_avx2_w4:
	CMPL CX, $4
	JL smexp_avx2_scalar
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
	JMP smexp_avx2_w4
smexp_avx2_scalar:
	TESTL CX, CX
	JZ smexp_avx2_fold
smexp_avx2_sloop:
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
	JNZ smexp_avx2_sloop
smexp_avx2_fold:
	VHADDPS Y5, Y5, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X0, X1, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func scaleF32AVX2(dst, src *float32, scale float32, count int)
TEXT ·scaleF32AVX2(SB), NOSPLIT, $0-16
	MOVL dst+0(FP), DI
	MOVL src+4(FP), SI
	VMOVSS scale+8(FP), X8
	VBROADCASTSS X8, Y8
	MOVL count+12(FP), CX
smscale_avx2_w8:
	CMPL CX, $8
	JL smscale_avx2_w4
	VMOVUPS (SI), Y0
	VMULPS Y8, Y0, Y0
	VMOVUPS Y0, (DI)
	ADDL $32, SI
	ADDL $32, DI
	SUBL $8, CX
	JMP smscale_avx2_w8
smscale_avx2_w4:
	CMPL CX, $4
	JL smscale_avx2_scalar
	VMOVUPS (SI), X0
	VMULPS X8, X0, X0
	VMOVUPS X0, (DI)
	ADDL $16, SI
	ADDL $16, DI
	SUBL $4, CX
	JMP smscale_avx2_w4
smscale_avx2_scalar:
	TESTL CX, CX
	JZ smscale_avx2_done
smscale_avx2_sloop:
	MOVSS (SI), X0
	VMULPS X8, X0, X0
	MOVSS X0, (DI)
	ADDL $4, SI
	ADDL $4, DI
	DECL CX
	JNZ smscale_avx2_sloop
smscale_avx2_done:
	RET

// func logSoftmaxShiftF32AVX2(dst, src *float32, maxValue, logSum float32, count int)
TEXT ·logSoftmaxShiftF32AVX2(SB), NOSPLIT, $0-20
	MOVL dst+0(FP), DI
	MOVL src+4(FP), SI
	VMOVSS maxValue+8(FP), X8
	VBROADCASTSS X8, Y8
	VMOVSS logSum+12(FP), X9
	VBROADCASTSS X9, Y9
	MOVL count+16(FP), CX
smlog_avx2_w8:
	CMPL CX, $8
	JL smlog_avx2_w4
	VMOVUPS (SI), Y0
	VSUBPS Y8, Y0, Y0
	VSUBPS Y9, Y0, Y7
	VMOVUPS Y7, (DI)
	ADDL $32, SI
	ADDL $32, DI
	SUBL $8, CX
	JMP smlog_avx2_w8
smlog_avx2_w4:
	CMPL CX, $4
	JL smlog_avx2_scalar
	VMOVUPS (SI), X0
	VSUBPS X8, X0, X0
	VSUBPS X9, X0, X7
	VMOVUPS X7, (DI)
	ADDL $16, SI
	ADDL $16, DI
	SUBL $4, CX
	JMP smlog_avx2_w4
smlog_avx2_scalar:
	TESTL CX, CX
	JZ smlog_avx2_done
smlog_avx2_sloop:
	MOVSS (SI), X0
	VSUBPS X8, X0, X0
	VSUBPS X9, X0, X7
	MOVSS X7, (DI)
	ADDL $4, SI
	ADDL $4, DI
	DECL CX
	JNZ smlog_avx2_sloop
smlog_avx2_done:
	RET

// func reduceMaxSoftmaxF32AVX2(src *float32, count int) float32
TEXT ·reduceMaxSoftmaxF32AVX2(SB), NOSPLIT, $0-8
	MOVL src+0(FP), SI
	MOVL count+4(FP), CX
	TESTL CX, CX
	JZ reduce_max_avx2_zero
	MOVSS (SI), X0
	VBROADCASTSS X0, Y0
	ADDL $4, SI
	DECL CX
reduce_max_avx2_w8:
	CMPL CX, $8
	JL reduce_max_avx2_w4
	VMOVUPS (SI), Y1
	VMAXPS Y1, Y0, Y0
	ADDL $32, SI
	SUBL $8, CX
	JMP reduce_max_avx2_w8
reduce_max_avx2_w4:
	CMPL CX, $4
	JL reduce_max_avx2_extract
	VMOVUPS (SI), X1
	VMAXPS X1, X0, X0
	ADDL $16, SI
	SUBL $4, CX
	JMP reduce_max_avx2_w4
reduce_max_avx2_extract:
	VEXTRACTF128 $1, Y0, X1
	VMAXPS X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
reduce_max_avx2_tail:
	TESTL CX, CX
	JZ reduce_max_avx2_done
reduce_max_avx2_sloop:
	MAXSS (SI), X0
	ADDL $4, SI
	DECL CX
	JNZ reduce_max_avx2_sloop
	JMP reduce_max_avx2_done
reduce_max_avx2_done:
	MOVSS X0, ret+8(FP)
	RET
reduce_max_avx2_zero:
	XORPS X0, X0
	MOVSS X0, ret+8(FP)
	RET

// func SoftmaxF32AVX2(dst, src *float32, count int)
TEXT ·SoftmaxF32AVX2(SB), NOSPLIT, $32-12
	MOVL dst+0(FP), BX
	MOVL src+4(FP), SI
	MOVL count+8(FP), CX
	MOVL SI, 0(SP)
	MOVL CX, 4(SP)
	CALL ·reduceMaxSoftmaxF32AVX2(SB)
	MOVSS X0, X6
	MOVL BX, 0(SP)
	MOVL SI, 4(SP)
	MOVSS X6, 8(SP)
	MOVL CX, 12(SP)
	CALL ·softmaxExpSumF32AVX2(SB)
	VMOVSS actX86LogC<>+4(SB), X1
	DIVSS X0, X1
	MOVL BX, 0(SP)
	MOVL BX, 4(SP)
	MOVSS X1, 8(SP)
	MOVL CX, 12(SP)
	CALL ·scaleF32AVX2(SB)
	RET

// func LogSoftmaxF32AVX2(dst, src *float32, count int)
TEXT ·LogSoftmaxF32AVX2(SB), NOSPLIT, $32-12
	MOVL dst+0(FP), BX
	MOVL src+4(FP), SI
	MOVL count+8(FP), CX
	MOVL SI, 0(SP)
	MOVL CX, 4(SP)
	CALL ·reduceMaxSoftmaxF32AVX2(SB)
	MOVSS X0, X6
	MOVL BX, 0(SP)
	MOVL SI, 4(SP)
	MOVSS X6, 8(SP)
	MOVL CX, 12(SP)
	CALL ·softmaxExpSumF32AVX2(SB)
	MOVSS X0, 0(SP)
	LEAL 0(SP), AX
	MOVL AX, 0(SP)
	MOVL AX, 4(SP)
	MOVL $1, 8(SP)
	CALL ·LogF32AVX2(SB)
	MOVSS X0, X7
	MOVL BX, 0(SP)
	MOVL SI, 4(SP)
	MOVSS X6, 8(SP)
	MOVSS X7, 12(SP)
	MOVL CX, 16(SP)
	CALL ·logSoftmaxShiftF32AVX2(SB)
	RET
