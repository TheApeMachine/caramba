#include "textflag.h"

// func ConvPatchDotFloat32AVX512Asm(weight, patch *float32, length int) float32
//
// Patch dot for conv2d/conv3d: sum(float64(weight[i])*float64(patch[i])) narrowed
// to float32. Hot loops widen f32 quads to f64 before multiply-accumulate.
TEXT ·ConvPatchDotFloat32AVX512Asm(SB), NOSPLIT, $0-28
	MOVQ weight+0(FP), SI
	MOVQ patch+8(FP), DI
	MOVQ length+16(FP), CX

	TESTQ CX, CX
	JZ   cpd_avx512_zero

	VXORPD Y0, Y0, Y0

cpd_avx512_w8:
	CMPQ CX, $8
	JL   cpd_avx512_w4

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
	JMP  cpd_avx512_w8

cpd_avx512_w4:
	CMPQ CX, $4
	JL   cpd_avx512_w4_tail

	VMOVUPS X1, (SI)
	VMOVUPS X2, (DI)
	VCVTPS2PD X1, Y5
	VCVTPS2PD X2, Y6
	VFMADD231PD Y0, Y6, Y5

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  cpd_avx512_w4

cpd_avx512_w4_tail:
	TESTQ CX, CX
	JZ   cpd_avx512_reduce

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7
	VMOVDQU32 (SI), K7, Y1
	VMOVDQU32 (DI), K7, Y2
	VEXTRACTF128 $0, Y1, X3
	VEXTRACTF128 $0, Y2, X4
	VCVTPS2PD X3, Y5
	VCVTPS2PD X4, Y6
	VFMADD231PD Y6, Y5, K7, Y0

cpd_avx512_reduce:
	VHADDPD Y1, Y0, Y0
	VHADDPD Y1, Y1, Y1
	VEXTRACTF128 $0, Y1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+24(FP)
	RET

cpd_avx512_zero:
	XORPS X0, X0
	MOVSS X0, ret+24(FP)
	RET
