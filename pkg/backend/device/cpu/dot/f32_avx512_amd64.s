#include "textflag.h"

// func DotFloat32AVX512Asm(left, right *float32, count int) float32
//
// Scalar reference is sum(float64(left[i])*float64(right[i])) narrowed to
// float32 once. Hot loops widen f32 quads to f64 before multiply and
// accumulate with VFMADD231PD into ymm lane partials, then fold.
TEXT ·DotFloat32AVX512Asm(SB), NOSPLIT, $0-28
	MOVQ left+0(FP), SI
	MOVQ right+8(FP), DI
	MOVQ count+16(FP), CX

	TESTQ CX, CX
	JZ   dot_zero

	VXORPD Y0, Y0, Y0

dot_w8:
	CMPQ CX, $8
	JL   dot_w4

	VMOVUPS Y1, (SI)
	VMOVUPS Y2, (DI)
	VEXTRACTF128 $0, Y1, X3
	VEXTRACTF128 $0, Y2, X4
	VCVTPS2PD X3, Y5
	VCVTPS2PD X4, Y6
	VFMADD231PD Y0, Y6, Y5
	VEXTRACTF128 $1, Y1, X3
	VEXTRACTF128 $1, Y2, X4
	VCVTPS2PD X3, Y5
	VCVTPS2PD X4, Y6
	VFMADD231PD Y0, Y6, Y5

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  dot_w8

dot_w4:
	CMPQ CX, $4
	JL   dot_w4_tail

	VMOVUPS X1, (SI)
	VMOVUPS X2, (DI)
	VCVTPS2PD X1, Y5
	VCVTPS2PD X2, Y6
	VFMADD231PD Y0, Y6, Y5

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  dot_w4

dot_w4_tail:
	TESTQ CX, CX
	JZ   dot_reduce

	MOVQ $1, AX
	SHLQ CL, AX
	DECQ AX
	KMOVQ AX, K7
	VMOVDQU32 (SI), K7, Y1
	VMOVDQU32 (DI), K7, Y2
	VEXTRACTF128 $0, Y1, X3
	VEXTRACTF128 $0, Y2, X4
	VCVTPS2PD X3, Y5
	VCVTPS2PD X4, Y6
	VFMADD231PD Y6, Y5, K7, Y0

dot_reduce:
	VHADDPD Y1, Y0, Y0
	VHADDPD Y1, Y1, Y1
	VEXTRACTF128 $0, Y1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+24(FP)
	RET

dot_zero:
	XORPS X0, X0
	MOVSS X0, ret+24(FP)
	RET
