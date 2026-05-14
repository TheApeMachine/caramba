#include "textflag.h"

// adamaxStepSSE2(out, m, u, params, grads []float64,
//                beta1, oneMinusBeta1, beta2, lrT, eps float64)
//
// Pure SSE2 AdaMax update, two float64 lanes per iteration:
//   m[i] = beta1*m[i] + (1-beta1)*g[i]
//   u[i] = max(beta2*u[i], abs(g[i]))
//   out[i] = params[i] - lrT*m[i]/(u[i]+eps)
TEXT ·adamaxStepSSE2(SB), NOSPLIT, $0-160
	MOVQ out+0(FP), AX
	MOVQ m+24(FP), R8
	MOVQ u+48(FP), R9
	MOVQ params+72(FP), R10
	MOVQ grads+96(FP), R11
	MOVQ out_len+8(FP), CX

	MOVSD  beta1+120(FP), X8
	SHUFPD $0, X8, X8
	MOVSD  oneMinusBeta1+128(FP), X9
	SHUFPD $0, X9, X9
	MOVSD  beta2+136(FP), X10
	SHUFPD $0, X10, X10
	MOVSD  lrT+144(FP), X11
	SHUFPD $0, X11, X11
	MOVSD  eps+152(FP), X12
	SHUFPD $0, X12, X12
	MOVSD  ·adamaxAbsMask(SB), X13
	SHUFPD $0, X13, X13

	CMPQ CX, $2
	JL   adamax_sse2_tail

adamax_sse2_loop:
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVUPD (R10), X2
	MOVUPD (R11), X3

	MULPD  X8, X0
	MOVAPD X3, X5
	MULPD  X9, X5
	ADDPD  X5, X0
	MOVUPD X0, (R8)

	MULPD  X10, X1
	MOVAPD X3, X4
	ANDPD  X13, X4
	MAXPD  X4, X1
	MOVUPD X1, (R9)

	MOVAPD X1, X5
	ADDPD  X12, X5
	MOVAPD X0, X6
	DIVPD  X5, X6
	MULPD  X11, X6
	SUBPD  X6, X2
	MOVUPD X2, (AX)

	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	ADDQ $16, R11
	SUBQ $2, CX
	CMPQ CX, $2
	JGE  adamax_sse2_loop

adamax_sse2_tail:
	CMPQ CX, $0
	JLE  adamax_sse2_done

	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVSD (R10), X2
	MOVSD (R11), X3

	MULSD  X8, X0
	MOVAPD X3, X5
	MULSD  X9, X5
	ADDSD  X5, X0
	MOVSD  X0, (R8)

	MULSD  X10, X1
	MOVAPD X3, X4
	ANDPD  X13, X4
	MAXSD  X4, X1
	MOVSD  X1, (R9)

	MOVAPD X1, X5
	ADDSD  X12, X5
	MOVAPD X0, X6
	DIVSD  X5, X6
	MULSD  X11, X6
	SUBSD  X6, X2
	MOVSD  X2, (AX)

adamax_sse2_done:
	RET
