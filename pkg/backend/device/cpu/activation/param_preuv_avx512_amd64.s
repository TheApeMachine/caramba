// SPDX-License-Identifier: Apache-2.0
// AVX-512 PReLU with per-element slope vectors (count == slopeCount).
#include "textflag.h"

// func PReLUVF32AVX512(dst, src, slopes *float32, count int)
TEXT ·PReLUVF32AVX512(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ slopes+16(FP), R8
	MOVQ count+24(FP), CX
	VXORPS Z15, Z15, Z15
preuv_avx512_w16:
	CMPQ CX, $16
	JL preuv_avx512_w8
	VMOVUPS (SI), Z0
	VMOVUPS (R8), Z10
	VCMPPS $6, Z15, Z0, K1
	VMULPS Z10, Z0, Z4
	VBLENDMPS Z4, Z0, K1, Z7
	VMOVUPS Z7, (DI)
	ADDQ $64, SI
	ADDQ $64, DI
	ADDQ $64, R8
	SUBQ $16, CX
	JMP preuv_avx512_w16
preuv_avx512_w8:
	CMPQ CX, $8
	JL preuv_avx512_w4
	VMOVUPS (SI), Y0
	VMOVUPS (R8), Y10
	VCMPPS $6, Y15, Y0, K1
	VMULPS Y10, Y0, Y4
	VBLENDMPS Y4, Y0, K1, Y7
	VMOVUPS Y7, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $32, R8
	SUBQ $8, CX
	JMP preuv_avx512_w8
preuv_avx512_w4:
	CMPQ CX, $4
	JL preuv_avx512_scalar
	VMOVUPS (SI), X0
	VMOVUPS (R8), X10
	VCMPPS $6, X15, X0, K1
	VMULPS X10, X0, X4
	VBLENDMPS X4, X0, K1, X7
	VMOVUPS X7, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	ADDQ $16, R8
	SUBQ $4, CX
	JMP preuv_avx512_w4
preuv_avx512_scalar:
	TESTQ CX, CX
	JZ preuv_avx512_done
preuv_avx512_sloop:
	MOVSS (SI), X0
	MOVSS (R8), X10
	VCMPPS $6, X15, X0, K1
	VMULPS X10, X0, X4
	VBLENDMPS X4, X0, K1, X7
	MOVSS X7, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	ADDQ $4, R8
	DECQ CX
	JNZ preuv_avx512_sloop
preuv_avx512_done:
	RET
