#include "textflag.h"

// laplacianAxisSetSSE2(out, left, center, right []float64, invH2 float64)
// out[i] = (left[i] + right[i] - 2*center[i]) * invH2
// SSE2 width is 2 doubles per xmm register.
TEXT ·laplacianAxisSetSSE2(SB), NOSPLIT, $0-104
	MOVQ     out+0(FP), DI
	MOVQ     out_len+8(FP), CX
	MOVQ     left+24(FP), R8
	MOVQ     center+48(FP), R9
	MOVQ     right+72(FP), R10
	MOVSD    invH2+96(FP), X15
	UNPCKLPD X15, X15
	CMPQ     CX, $2
	JL       done_set_sse2
loop_set_sse2:
	MOVUPD (R8), X0
	MOVUPD (R10), X1
	ADDPD  X1, X0
	MOVUPD (R9), X2
	ADDPD  X2, X2
	SUBPD  X2, X0
	MULPD  X15, X0
	MOVUPD X0, (DI)
	ADDQ   $16, DI
	ADDQ   $16, R8
	ADDQ   $16, R9
	ADDQ   $16, R10
	SUBQ   $2, CX
	CMPQ   CX, $2
	JGE    loop_set_sse2
done_set_sse2:
	RET

// laplacianAxisAccSSE2(out, left, center, right []float64, invH2 float64)
// out[i] += (left[i] + right[i] - 2*center[i]) * invH2
TEXT ·laplacianAxisAccSSE2(SB), NOSPLIT, $0-104
	MOVQ     out+0(FP), DI
	MOVQ     out_len+8(FP), CX
	MOVQ     left+24(FP), R8
	MOVQ     center+48(FP), R9
	MOVQ     right+72(FP), R10
	MOVSD    invH2+96(FP), X15
	UNPCKLPD X15, X15
	CMPQ     CX, $2
	JL       done_acc_sse2
loop_acc_sse2:
	MOVUPD (R8), X0
	MOVUPD (R10), X1
	ADDPD  X1, X0
	MOVUPD (R9), X2
	ADDPD  X2, X2
	SUBPD  X2, X0
	MULPD  X15, X0
	MOVUPD (DI), X3
	ADDPD  X3, X0
	MOVUPD X0, (DI)
	ADDQ   $16, DI
	ADDQ   $16, R8
	ADDQ   $16, R9
	ADDQ   $16, R10
	SUBQ   $2, CX
	CMPQ   CX, $2
	JGE    loop_acc_sse2
done_acc_sse2:
	RET
