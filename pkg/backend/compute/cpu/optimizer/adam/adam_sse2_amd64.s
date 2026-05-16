#include "textflag.h"

// adamStepSSE2(out, mState, vState, params, grads []float64,
//              beta1, oneMinusBeta1, beta2, oneMinusBeta2,
//              lrT, eps, lrWD float64)
//
// Pure SSE2 Adam update, two float64 lanes per iteration:
//   m[i] = beta1*m[i] + (1-beta1)*g[i]
//   v[i] = beta2*v[i] + (1-beta2)*g[i]^2
//   out[i] = params[i]*(1-lrWD) - lrT*m[i]/(sqrt(v[i])+eps)
TEXT ·adamStepSSE2(SB), NOSPLIT, $0-176
	MOVQ out+0(FP), AX
	MOVQ m+24(FP), R8
	MOVQ v+48(FP), R9
	MOVQ params+72(FP), R10
	MOVQ grads+96(FP), R11
	MOVQ out_len+8(FP), CX

	MOVSD  beta1+120(FP), X8
	SHUFPD $0, X8, X8
	MOVSD  oneMinusBeta1+128(FP), X9
	SHUFPD $0, X9, X9
	MOVSD  beta2+136(FP), X10
	SHUFPD $0, X10, X10
	MOVSD  oneMinusBeta2+144(FP), X11
	SHUFPD $0, X11, X11
	MOVSD  lrT+152(FP), X12
	SHUFPD $0, X12, X12
	MOVSD  eps+160(FP), X13
	SHUFPD $0, X13, X13
	MOVSD  lrWD+168(FP), X14
	SHUFPD $0, X14, X14

	CMPQ CX, $2
	JL   adam_sse2_tail

adam_sse2_loop:
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVUPD (R10), X2
	MOVUPD (R11), X3

	MOVAPD X3, X4
	MULPD  X3, X4

	MULPD  X8, X0
	MOVAPD X3, X5
	MULPD  X9, X5
	ADDPD  X5, X0
	MOVUPD X0, (R8)

	MULPD  X10, X1
	MOVAPD X4, X5
	MULPD  X11, X5
	ADDPD  X5, X1
	MOVUPD X1, (R9)

	MOVAPD X1, X5
	SQRTPD X5, X5
	ADDPD  X13, X5

	MOVAPD X0, X6
	DIVPD  X5, X6

	MOVAPD X2, X7
	MOVAPD X2, X15
	MULPD  X14, X15
	SUBPD  X15, X7

	MULPD  X12, X6
	SUBPD  X6, X7
	MOVUPD X7, (AX)

	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	ADDQ $16, R11
	SUBQ $2, CX
	CMPQ CX, $2
	JGE  adam_sse2_loop

adam_sse2_tail:
	CMPQ CX, $0
	JLE  adam_sse2_done

	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVSD (R10), X2
	MOVSD (R11), X3

	MOVAPD X3, X4
	MULSD  X3, X4

	MULSD  X8, X0
	MOVAPD X3, X5
	MULSD  X9, X5
	ADDSD  X5, X0
	MOVSD  X0, (R8)

	MULSD  X10, X1
	MOVAPD X4, X5
	MULSD  X11, X5
	ADDSD  X5, X1
	MOVSD  X1, (R9)

	MOVAPD X1, X5
	SQRTSD X5, X5
	ADDSD  X13, X5

	MOVAPD X0, X6
	DIVSD  X5, X6

	MOVAPD X2, X7
	MOVAPD X2, X15
	MULSD  X14, X15
	SUBSD  X15, X7

	MULSD  X12, X6
	SUBSD  X6, X7
	MOVSD  X7, (AX)

adam_sse2_done:
	RET
