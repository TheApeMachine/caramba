// SPDX-License-Identifier: Apache-2.0
// AVX2 PReLU with per-element slope vectors (count == slopeCount).
#include "textflag.h"

// func PReLUVF32AVX2(dst, src, slopes *float32, count int)
TEXT ·PReLUVF32AVX2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ slopes+16(FP), R8
	MOVQ count+24(FP), CX
	VXORPS Y15, Y15, Y15
preuv_avx2_w8:
	CMPQ CX, $8
	JL preuv_avx2_w4
	VMOVUPS (SI), Y0
	VMOVUPS (R8), Y10
	VCMPPS $6, Y15, Y0, Y2
	VMULPS Y10, Y0, Y4
	VANDPS Y2, Y0, Y3
	VPANDN Y2, Y4, Y4
	VORPS Y3, Y4, Y7
	VMOVUPS Y7, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $32, R8
	SUBQ $8, CX
	JMP preuv_avx2_w8
preuv_avx2_w4:
	CMPQ CX, $4
	JL preuv_avx2_scalar
	VMOVUPS (SI), X0
	VMOVUPS (R8), X10
	VCMPPS $6, X15, X0, X2
	VMULPS X10, X0, X4
	VANDPS X2, X0, X3
	VPANDN X2, X4, X4
	VORPS X3, X4, X7
	VMOVUPS X7, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	ADDQ $16, R8
	SUBQ $4, CX
	JMP preuv_avx2_w4
preuv_avx2_scalar:
	TESTQ CX, CX
	JZ preuv_avx2_done
preuv_avx2_sloop:
	MOVSS (SI), X0
	MOVSS (R8), X10
	VCMPPS $6, X15, X0, X2
	VMULPS X10, X0, X4
	VANDPS X2, X0, X3
	VPANDN X2, X4, X4
	VORPS X3, X4, X7
	MOVSS X7, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	ADDQ $4, R8
	DECQ CX
	JNZ preuv_avx2_sloop
preuv_avx2_done:
	RET
