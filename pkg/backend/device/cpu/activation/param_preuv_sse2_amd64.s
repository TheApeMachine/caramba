// SPDX-License-Identifier: Apache-2.0
// SSE2 PReLU with per-element slope vectors (count == slopeCount).
#include "textflag.h"

// func PReLUVF32SSE2(dst, src, slopes *float32, count int)
TEXT ·PReLUVF32SSE2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ slopes+16(FP), R8
	MOVQ count+24(FP), CX
	XORPS X15, X15
preuv_sse2_w4:
	CMPQ CX, $4
	JL preuv_sse2_done
	VMOVUPS (SI), X0
	VMOVUPS (R8), X10
	VCMPPS $6, X15, X0, X2
	VMULPS X10, X0, X4
	VANDPS X2, X0, X3
	VANDNPS X4, X2, X2
	VORPS X3, X2, X7
	VMOVUPS X7, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	ADDQ $16, R8
	SUBQ $4, CX
	JMP preuv_sse2_w4
preuv_sse2_done:
	RET
