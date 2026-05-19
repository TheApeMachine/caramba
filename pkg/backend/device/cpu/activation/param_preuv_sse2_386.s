// SPDX-License-Identifier: Apache-2.0
// 386 SSE2 PReLU with per-element slope vectors (count == slopeCount).
#include "textflag.h"

// func PReLUVF32SSE2(dst, src, slopes *float32, count int)
TEXT ·PReLUVF32SSE2(SB), NOSPLIT, $0-16
	MOVL dst+0(FP), DI
	MOVL src+4(FP), SI
	MOVL slopes+8(FP), BX
	MOVL count+12(FP), CX
	VXORPS X3, X3, X3
preuv_sse2_386_w4:
	CMPL CX, $4
	JL preuv_sse2_386_scalar
	VMOVUPS (SI), X0
	VMOVUPS (BX), X6
	VPXOR X3, X3, X3
	VPCMPGTD X3, X0, X2
	VMULPS X6, X0, X4
	VANDPS X2, X0, X1
	VPANDN X2, X4, X4
	VORPS X1, X4, X5
	VMOVUPS X5, (DI)
	ADDL $16, SI
	ADDL $16, DI
	ADDL $16, BX
	SUBL $4, CX
	JMP preuv_sse2_386_w4
preuv_sse2_386_scalar:
	TESTL CX, CX
	JZ preuv_sse2_386_done
preuv_sse2_386_sloop:
	VMOVSS (SI), X0
	VMOVSS (BX), X6
	VPXOR X3, X3, X3
	VPCMPGTD X3, X0, X2
	VMULPS X6, X0, X4
	VANDPS X2, X0, X1
	VPANDN X2, X4, X4
	VORPS X1, X4, X5
	MOVSS X5, (DI)
	ADDL $4, SI
	ADDL $4, DI
	ADDL $4, BX
	DECL CX
	JNZ preuv_sse2_386_sloop
preuv_sse2_386_done:
	RET
