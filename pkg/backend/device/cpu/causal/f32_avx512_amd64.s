// SPDX-License-Identifier: Apache-2.0
// AVX-512 float32 causal kernels: CATE subtract, counterfactual, strided dot.
#include "textflag.h"

// func CateFloat32AVX512Asm(treated, control, out *float32, count int)
TEXT ·CateFloat32AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ treated+0(FP), DI
	MOVQ control+8(FP), SI
	MOVQ out+16(FP), R8
	MOVQ count+24(FP), CX

cate_w16:
	CMPQ CX, $16
	JL   cate_w8

	VMOVUPS Y0, (DI)
	VMOVUPS Y1, (SI)
	VSUBPS  Y1, Y0, Y0
	VMOVUPS Y0, (R8)
	VMOVUPS Y2, 32(DI)
	VMOVUPS Y3, 32(SI)
	VSUBPS  Y3, Y2, Y2
	VMOVUPS Y2, 32(R8)

	ADDQ $64, DI
	ADDQ $64, SI
	ADDQ $64, R8
	SUBQ $16, CX
	JMP  cate_w16

cate_w8:
	CMPQ CX, $8
	JL   cate_w4

	VMOVUPS Y0, (DI)
	VMOVUPS Y1, (SI)
	VSUBPS  Y1, Y0, Y0
	VMOVUPS Y0, (R8)

	ADDQ $32, DI
	ADDQ $32, SI
	ADDQ $32, R8
	SUBQ $8, CX
	JMP  cate_w8

cate_w4:
	CMPQ CX, $4
	JL   cate_w4_tail

	VMOVUPS X0, (DI)
	VMOVUPS X1, (SI)
	VSUBPS  X1, X0, X0
	VMOVUPS X0, (R8)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	SUBQ $4, CX
	JMP  cate_w4

cate_w4_tail:
	TESTQ CX, CX
	JZ   cate_done

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (DI), K7, Y0
	VMOVDQU32 (SI), K7, Y1
	VSUBPS  Y1, Y0, Y0
	VMOVDQU32 Y0, K7, (R8)

cate_done:
	RET

// func CounterfactualFloat32AVX512Asm(out, observedY, observedX, counterfactualX *float32, slope float32, count int)
TEXT ·CounterfactualFloat32AVX512Asm(SB), NOSPLIT, $0-48
	MOVQ out+0(FP), DI
	MOVQ observedY+8(FP), SI
	MOVQ observedX+16(FP), R9
	MOVQ counterfactualX+24(FP), R10
	MOVSS slope+32(FP), X15
	MOVQ count+40(FP), CX
	VSHUFPS $0, X15, X15, X15
	VBROADCASTSS X15, Y15

cf_w16:
	CMPQ CX, $16
	JL   cf_w8

	VMOVUPS Y0, (R10)
	VMOVUPS Y1, (R9)
	VSUBPS  Y1, Y0, Y0
	VMULPS  Y15, Y0, Y0
	VMOVUPS Y2, (SI)
	VADDPS  Y0, Y2, Y0
	VMOVUPS Y0, (DI)
	VMOVUPS Y3, 32(R10)
	VMOVUPS Y4, 32(R9)
	VSUBPS  Y4, Y3, Y3
	VMULPS  Y15, Y3, Y3
	VMOVUPS Y5, 32(SI)
	VADDPS  Y3, Y5, Y3
	VMOVUPS Y3, 32(DI)

	ADDQ $64, DI
	ADDQ $64, SI
	ADDQ $64, R9
	ADDQ $64, R10
	SUBQ $16, CX
	JMP  cf_w16

cf_w8:
	CMPQ CX, $8
	JL   cf_w4

	VMOVUPS Y0, (R10)
	VMOVUPS Y1, (R9)
	VSUBPS  Y1, Y0, Y0
	VMULPS  Y15, Y0, Y0
	VMOVUPS Y2, (SI)
	VADDPS  Y0, Y2, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, DI
	ADDQ $32, SI
	ADDQ $32, R9
	ADDQ $32, R10
	SUBQ $8, CX
	JMP  cf_w8

cf_w4:
	CMPQ CX, $4
	JL   cf_w4_tail

	VMOVUPS X0, (R10)
	VMOVUPS X1, (R9)
	VSUBPS  X1, X0, X0
	VMULPS  X15, X0, X0
	VMOVUPS X2, (SI)
	VADDPS  X0, X2, X0
	VMOVUPS X0, (DI)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $4, CX
	JMP  cf_w4

cf_w4_tail:
	TESTQ CX, CX
	JZ   cf_done

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (R10), K7, Y0
	VMOVDQU32 (R9), K7, Y1
	VSUBPS  Y1, Y0, Y0
	VMULPS  Y15, Y0, Y0
	VMOVDQU32 (SI), K7, Y2
	VADDPS  Y0, Y2, Y0
	VMOVDQU32 Y0, K7, (DI)

cf_done:
	RET

// func StridedDotFloat32AVX512Asm(values *float32, stride int, weights *float32, count int) float32
//
// Hot path: dword index vector × stride, byte offsets via VPSLLD, VGATHERDPS loads,
// contiguous weight VMOVUPS, f64 widen-multiply-accumulate (dot package contract).
TEXT ·StridedDotFloat32AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ values+0(FP), SI
	MOVQ stride+8(FP), R9
	MOVQ weights+16(FP), DI
	MOVQ count+24(FP), CX

	TESTQ CX, CX
	JZ   strided_zero

	MOVQ R9, R10
	SHLQ $2, R10

	VXORPD Y0, Y0, Y0
	KXNORW K1, K1, K1
	VPBROADCASTD R9, K1, Y5

strided_w8:
	CMPQ CX, $8
	JL   strided_w4

	VMOVDQA32 ·stridedDotIota8<>(SB), Y6
	VPMULLD Y5, Y6, Y6
	VPSLLD  $2, Y6, Y6
	VGATHERDPS (SI)(Y6*1), K1, Y7
	VMOVUPS Y8, (DI)

	VEXTRACTF128 $0, Y7, X1
	VEXTRACTF128 $0, Y8, X2
	VCVTPS2PD X1, Y10
	VCVTPS2PD X2, Y11
	VFMADD231PD Y0, Y11, Y10

	VEXTRACTF128 $1, Y7, X1
	VEXTRACTF128 $1, Y8, X2
	VCVTPS2PD X1, Y10
	VCVTPS2PD X2, Y11
	VFMADD231PD Y0, Y11, Y10

	ADDQ $32, DI
	MOVQ R10, AX
	SHLQ $3, AX
	ADDQ AX, SI
	SUBQ $8, CX
	JMP  strided_w8

strided_w4:
	CMPQ CX, $4
	JL   strided_w4_tail

	VMOVDQA32 ·stridedDotIota4<>(SB), Y6
	VPMULLD Y5, Y6, Y6
	VPSLLD  $2, Y6, Y6
	VGATHERDPS (SI)(Y6*1), K1, Y7
	VMOVUPS X8, (DI)

	VCVTPS2PD X7, Y10
	VCVTPS2PD X8, Y11
	VFMADD231PD Y0, Y11, Y10

	ADDQ $16, DI
	MOVQ R10, AX
	SHLQ $2, AX
	ADDQ AX, SI
	SUBQ $4, CX
	JMP  strided_w4

strided_w4_tail:
	TESTQ CX, CX
	JZ   strided_reduce

	MOVQ $1, AX
	MOVQ CX, DX
	SHLQ CL, AX
	DECQ AX
	KMOVQ AX, K7

	VMOVDQA32 ·stridedDotIota8<>(SB), Y6
	VPMULLD Y5, Y6, Y6
	VPSLLD  $2, Y6, Y6
	VGATHERDPS (SI)(Y6*1), K7, Y7
	VMOVDQU32 (DI), K7, Y8

	VEXTRACTF128 $0, Y7, X1
	VEXTRACTF128 $0, Y8, X2
	VCVTPS2PD X1, Y10
	VCVTPS2PD X2, Y11
	VFMADD231PD Y11, Y10, K7, Y0

strided_reduce:
	VHADDPD Y1, Y0, Y0
	VHADDPD Y1, Y1, Y1
	VEXTRACTF128 $0, Y1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+32(FP)
	RET

strided_zero:
	XORPS X0, X0
	MOVSS X0, ret+32(FP)
	RET

DATA ·stridedDotIota4<>+0(SB)/4, $0
DATA ·stridedDotIota4<>+4(SB)/4, $1
DATA ·stridedDotIota4<>+8(SB)/4, $2
DATA ·stridedDotIota4<>+12(SB)/4, $3
GLOBL ·stridedDotIota4<>(SB), RODATA, $16

DATA ·stridedDotIota8<>+0(SB)/4, $0
DATA ·stridedDotIota8<>+4(SB)/4, $1
DATA ·stridedDotIota8<>+8(SB)/4, $2
DATA ·stridedDotIota8<>+12(SB)/4, $3
DATA ·stridedDotIota8<>+16(SB)/4, $4
DATA ·stridedDotIota8<>+20(SB)/4, $5
DATA ·stridedDotIota8<>+24(SB)/4, $6
DATA ·stridedDotIota8<>+28(SB)/4, $7
GLOBL ·stridedDotIota8<>(SB), RODATA, $32
