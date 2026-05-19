// SPDX-License-Identifier: Apache-2.0
// 386 SSE2 PReLU with per-element slope vectors (count == slopeCount).
#include "textflag.h"

// func PReLUVF32SSE2(dst, src, slopes *float32, count int)
TEXT ·PReLUVF32SSE2(SB), NOSPLIT, $0-16
	MOVL dst+0(FP), DI
	MOVL src+4(FP), SI
	MOVL slopes+8(FP), BX
	MOVL count+12(FP), CX
	XORPS X3, X3
preuv_sse2_386_w4:
	CMPL CX, $4
	JL preuv_sse2_386_scalar
	MOVUPS (SI), X0
	MOVUPS (BX), X6
	MOVAPS X0, X2
	XORPS X3, X3
	CMPPS $6, X3, X2
	MOVAPS X0, X4
	MULPS X6, X4
	MOVAPS X0, X1
	ANDPS X2, X1
	ANDNPS X4, X2
	MOVAPS X1, X5
	ORPS X2, X5
	MOVUPS X5, (DI)
	ADDL $16, SI
	ADDL $16, DI
	ADDL $16, BX
	SUBL $4, CX
	JMP preuv_sse2_386_w4
preuv_sse2_386_scalar:
	TESTL CX, CX
	JZ preuv_sse2_386_done
preuv_sse2_386_sloop:
	MOVSS (SI), X0
	MOVSS (BX), X6
	MOVAPS X0, X2
	XORPS X3, X3
	CMPSS $6, X3, X2
	MOVAPS X0, X4
	MULSS X6, X4
	MOVAPS X0, X1
	ANDPS X2, X1
	ANDNPS X4, X2
	MOVAPS X1, X5
	ORPS X2, X5
	MOVSS X5, (DI)
	ADDL $4, SI
	ADDL $4, DI
	ADDL $4, BX
	DECL CX
	JNZ preuv_sse2_386_sloop
preuv_sse2_386_done:
	RET
