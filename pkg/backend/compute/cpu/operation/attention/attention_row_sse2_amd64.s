#include "textflag.h"

// attentionRowScoresSSE2(scores, q, K []float64, seqLen int, headDim int, scale float64)
//   scores[j] = scale * sum(q[d] * K[j*headDim+d]).
TEXT ·attentionRowScoresSSE2(SB), NOSPLIT, $0-96
	MOVQ scores+0(FP), AX
	MOVQ q+24(FP), R8
	MOVQ K+48(FP), R9
	MOVQ seqLen+72(FP), R10
	MOVQ headDim+80(FP), R11
	MOVSD scale+88(FP), X15
	SHUFPD $0, X15, X15

	XORQ R12, R12
arss_j:
	CMPQ R12, R10
	JGE arss_done
	MOVQ R12, BX
	IMULQ R11, BX
	SHLQ $3, BX
	MOVQ R9, SI
	ADDQ BX, SI
	MOVQ R8, DI
	MOVQ R11, CX
	XORPD X0, X0
	CMPQ CX, $2
	JL arss_dot_tail
arss_dot_loop:
	MOVUPD (DI), X1
	MOVUPD (SI), X2
	MULPD X2, X1
	ADDPD X1, X0
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, CX
	CMPQ CX, $2
	JGE arss_dot_loop
	MOVAPD X0, X1
	UNPCKHPD X0, X1
	ADDSD X1, X0
arss_dot_tail:
	CMPQ CX, $0
	JLE arss_dot_done
	MOVSD (DI), X1
	MOVSD (SI), X2
	MULSD X2, X1
	ADDSD X1, X0
arss_dot_done:
	MULSD X15, X0
	MOVQ R12, BX
	SHLQ $3, BX
	MOVQ AX, SI
	ADDQ BX, SI
	MOVSD X0, (SI)
	INCQ R12
	JMP arss_j
arss_done:
	RET

// attentionRowOutputSSE2(out, scores, V []float64, seqLen int, headDim int)
//   out[d] = sum(scores[j] * V[j*headDim+d]).
TEXT ·attentionRowOutputSSE2(SB), NOSPLIT, $0-88
	MOVQ out+0(FP), AX
	MOVQ scores+24(FP), R8
	MOVQ V+48(FP), R9
	MOVQ seqLen+72(FP), R10
	MOVQ headDim+80(FP), R11

	MOVQ R11, CX
	MOVQ AX, SI
	XORPD X0, X0
aros_clear:
	CMPQ CX, $2
	JL aros_clear_scalar
	MOVUPD X0, (SI)
	ADDQ $16, SI
	SUBQ $2, CX
	JMP aros_clear
aros_clear_scalar:
	CMPQ CX, $0
	JLE aros_clear_done
	MOVSD X0, (SI)
aros_clear_done:

	XORQ R12, R12
aros_j:
	CMPQ R12, R10
	JGE aros_done
	MOVQ R12, BX
	IMULQ R11, BX
	SHLQ $3, BX
	MOVQ R9, SI
	ADDQ BX, SI
	MOVQ AX, DI
	MOVQ R12, BX
	SHLQ $3, BX
	MOVQ R8, DX
	ADDQ BX, DX
	MOVSD (DX), X14
	SHUFPD $0, X14, X14
	MOVQ R11, CX
	CMPQ CX, $2
	JL aros_w_scalar
aros_w_loop:
	MOVUPD (DI), X0
	MOVUPD (SI), X1
	MOVAPD X14, X2
	MULPD X1, X2
	ADDPD X2, X0
	MOVUPD X0, (DI)
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, CX
	CMPQ CX, $2
	JGE aros_w_loop
aros_w_scalar:
	CMPQ CX, $0
	JLE aros_w_done
	MOVSD (DI), X0
	MOVSD (SI), X1
	MULSD X14, X1
	ADDSD X1, X0
	MOVSD X0, (DI)
aros_w_done:
	INCQ R12
	JMP aros_j
aros_done:
	RET
