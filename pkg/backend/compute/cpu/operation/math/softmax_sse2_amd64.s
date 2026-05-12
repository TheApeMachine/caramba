#include "textflag.h"

// softmaxRowSSE2(row []float64)
// 2-lane SSE2 variant. Uses exp constants from exp_avx2_amd64.s.
TEXT ·softmaxRowSSE2(SB), NOSPLIT, $0-24
	MOVQ row+0(FP), AX
	MOVQ row_len+8(FP), CX
	CMPQ CX, $0
	JLE sms_done
	MOVQ CX, R11
	MOVQ AX, R12

	// Pass 1: find max
	MOVSD (AX), X0
	MOVQ AX, R14
	MOVQ CX, R13
sms_max_loop:
	MOVSD (R14), X1
	MAXSD X1, X0
	ADDQ $8, R14
	DECQ R13
	JNZ sms_max_loop

	MOVAPD X0, X10                              // scalar max

	// Pass 2: exp(row[i] - max) per element using SSE2 polynomial
	MOVQ R11, CX
	MOVQ R12, AX
sms_exp_loop:
	MOVSD (AX), X0
	SUBSD X10, X0
	MOVSD ·expMaxArg(SB), X1
	MINSD X1, X0
	MOVSD ·expMinArg(SB), X1
	MAXSD X1, X0

	MOVSD ·expLog2E(SB), X1
	MULSD X0, X1
	CVTSD2SQ X1, BX                             // r as int64
	CVTSQ2SD BX, X1                             // round-trip = round-to-nearest

	MOVSD ·expLn2Hi(SB), X2
	MULSD X1, X2
	SUBSD X2, X0
	MOVSD ·expLn2Lo(SB), X2
	MULSD X1, X2
	SUBSD X2, X0

	MOVSD ·expC11(SB), X2
	MULSD X0, X2
	ADDSD ·expC10(SB), X2
	MULSD X0, X2
	ADDSD ·expC9(SB), X2
	MULSD X0, X2
	ADDSD ·expC8(SB), X2
	MULSD X0, X2
	ADDSD ·expC7(SB), X2
	MULSD X0, X2
	ADDSD ·expC6(SB), X2
	MULSD X0, X2
	ADDSD ·expC5(SB), X2
	MULSD X0, X2
	ADDSD ·expC4(SB), X2
	MULSD X0, X2
	ADDSD ·expC3(SB), X2
	MULSD X0, X2
	ADDSD ·expC2(SB), X2
	MULSD X0, X2
	ADDSD ·expC1(SB), X2
	MULSD X0, X2
	ADDSD ·expC0(SB), X2

	ADDQ $1023, BX
	SHLQ $52, BX
	MOVQ BX, X3
	MULSD X3, X2

	MOVSD X2, (AX)
	ADDQ $8, AX
	DECQ CX
	JNZ sms_exp_loop

	// Pass 3: sum
	XORPD X11, X11
	MOVQ R11, CX
	MOVQ R12, AX
sms_sum_loop:
	MOVSD (AX), X0
	ADDSD X0, X11
	ADDQ $8, AX
	DECQ CX
	JNZ sms_sum_loop

	// Pass 4: divide
	MOVSD $1.0, X12
	DIVSD X11, X12
	MOVQ R11, CX
	MOVQ R12, AX
sms_div_loop:
	MOVSD (AX), X0
	MULSD X12, X0
	MOVSD X0, (AX)
	ADDQ $8, AX
	DECQ CX
	JNZ sms_div_loop

sms_done:
	RET
