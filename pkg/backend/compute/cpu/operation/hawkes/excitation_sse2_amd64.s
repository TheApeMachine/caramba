#include "textflag.h"

// hawkesExcitationSSE2(events []float64, now, beta, alpha float64) float64
// returns alpha * sum(exp(-beta*(now-events[i]))).
TEXT ·hawkesExcitationSSE2(SB), NOSPLIT, $0-56
	MOVQ events+0(FP), AX
	MOVQ events_len+8(FP), CX
	CMPQ CX, $0
	JLE  hexs_done
	XORPD X14, X14

hexs_loop:
	MOVSD (AX), X0
	MOVSD now+24(FP), X1
	SUBSD X0, X1
	MOVSD beta+32(FP), X2
	XORPD X3, X3
	SUBSD X2, X3
	MULSD X3, X1
	MOVAPD X1, X0

	MOVSD ·hexMaxArg(SB), X1
	MINSD X1, X0
	MOVSD ·hexMinArg(SB), X1
	MAXSD X1, X0

	MOVSD ·hexLog2E(SB), X1
	MULSD X0, X1
	ROUNDSD $0, X1, X1
	CVTTSD2SQ X1, BX

	MOVSD ·hexLn2Hi(SB), X2
	MULSD X1, X2
	SUBSD X2, X0
	MOVSD ·hexLn2Lo(SB), X2
	MULSD X1, X2
	SUBSD X2, X0

	MOVSD ·hexC11(SB), X2
	MULSD X0, X2
	ADDSD ·hexC10(SB), X2
	MULSD X0, X2
	ADDSD ·hexC9(SB), X2
	MULSD X0, X2
	ADDSD ·hexC8(SB), X2
	MULSD X0, X2
	ADDSD ·hexC7(SB), X2
	MULSD X0, X2
	ADDSD ·hexC6(SB), X2
	MULSD X0, X2
	ADDSD ·hexC5(SB), X2
	MULSD X0, X2
	ADDSD ·hexC4(SB), X2
	MULSD X0, X2
	ADDSD ·hexC3(SB), X2
	MULSD X0, X2
	ADDSD ·hexC2(SB), X2
	MULSD X0, X2
	ADDSD ·hexC1(SB), X2
	MULSD X0, X2
	ADDSD ·hexC0(SB), X2

	ADDQ $1023, BX
	SHLQ $52, BX
	MOVQ BX, X3
	MULSD X3, X2
	ADDSD X2, X14
	ADDQ  $8, AX
	DECQ  CX
	JNZ   hexs_loop

	MULSD alpha+40(FP), X14
	MOVSD X14, ret+48(FP)
	RET

hexs_done:
	XORPD X0, X0
	MOVSD X0, ret+48(FP)
	RET
