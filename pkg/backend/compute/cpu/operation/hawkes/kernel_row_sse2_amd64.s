#include "textflag.h"

// hawkesKernelRowSSE2(out, events []float64, ti, alpha, beta float64)
// writes out[i] = alpha * exp(-beta * (events[i] - ti)).
TEXT ·hawkesKernelRowSSE2(SB), NOSPLIT, $0-72
	MOVQ out+0(FP), AX
	MOVQ events+24(FP), DI
	MOVQ out_len+8(FP), CX
	CMPQ CX, $0
	JLE  hkrs_done

hkrs_loop:
	MOVSD (DI), X0
	MOVSD ti+48(FP), X1
	SUBSD X1, X0
	MOVSD beta+64(FP), X1
	XORPD X3, X3
	SUBSD X1, X3
	MULSD X3, X0

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
	MULSD alpha+56(FP), X2
	MOVSD X2, (AX)

	ADDQ $8, AX
	ADDQ $8, DI
	DECQ CX
	JNZ  hkrs_loop

hkrs_done:
	RET
