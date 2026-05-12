#include "textflag.h"

// softmaxRowNEON(row []float64)
// Numerically-stable in-place softmax (scalar-FP).
TEXT ·softmaxRowNEON(SB), NOSPLIT, $0-24
	MOVD row+0(FP), R0
	MOVD row_len+8(FP), R1
	CBZ R1, sm_done
	MOVD R1, R11                              // saved n
	MOVD R0, R12                              // saved ptr

	// Pass 1: find max
	FMOVD (R0), F0
	MOVD R0, R7
	MOVD R1, R8
sm_max_loop:
	FMOVD (R7), F1
	FMAXD F1, F0, F0
	ADD $8, R7, R7
	SUBS $1, R8, R8
	BNE sm_max_loop
	// F0 = max

	// Load exp constants
	FMOVD ·expLog2E(SB), F20
	FMOVD ·expLn2Hi(SB), F21
	FMOVD ·expLn2Lo(SB), F22
	FMOVD ·expMaxArg(SB), F23
	FMOVD ·expMinArg(SB), F24
	FMOVD ·expC11(SB), F25
	FMOVD ·expC10(SB), F26
	MOVD  $1023, R10

	// Pass 2: exp(row[i] - max)
	MOVD R12, R7
	MOVD R11, R8
sm_exp_loop:
	FMOVD (R7), F2
	FSUBD F0, F2, F2
	FMINNMD F23, F2, F2
	FMAXNMD F24, F2, F2

	FMULD F20, F2, F3
	FRINTND F3, F3
	FMSUBD F3, F2, F21, F2                     // x -= r*ln2_hi
	FMSUBD F3, F2, F22, F2                     // x -= r*ln2_lo

	// Polynomial degree 11
	FMOVD F25, F4
	FMADDD F4, F26, F2, F4
	FMOVD ·expC9(SB), F5
	FMADDD F4, F5, F2, F4
	FMOVD ·expC8(SB), F5
	FMADDD F4, F5, F2, F4
	FMOVD ·expC7(SB), F5
	FMADDD F4, F5, F2, F4
	FMOVD ·expC6(SB), F5
	FMADDD F4, F5, F2, F4
	FMOVD ·expC5(SB), F5
	FMADDD F4, F5, F2, F4
	FMOVD ·expC4(SB), F5
	FMADDD F4, F5, F2, F4
	FMOVD ·expC3(SB), F5
	FMADDD F4, F5, F2, F4
	FMOVD ·expC2(SB), F5
	FMADDD F4, F5, F2, F4
	FMOVD ·expC1(SB), F5
	FMADDD F4, F5, F2, F4
	FMOVD ·expC0(SB), F5
	FMADDD F4, F5, F2, F4                      // poly

	// 2^r
	FCVTZSD F3, R5
	ADD R10, R5, R5
	LSL $52, R5, R5
	FMOVD R5, F6
	FMULD F6, F4, F4

	FMOVD F4, (R7)
	ADD $8, R7, R7
	SUBS $1, R8, R8
	BNE sm_exp_loop

	// Pass 3: sum
	FMOVD $0.0, F7
	MOVD R12, R7
	MOVD R11, R8
sm_sum_loop:
	FMOVD (R7), F2
	FADDD F2, F7, F7
	ADD $8, R7, R7
	SUBS $1, R8, R8
	BNE sm_sum_loop

	// Pass 4: divide
	FMOVD $1.0, F8
	FDIVD F7, F8, F8                            // F8 = 1/sum
	MOVD R12, R7
	MOVD R11, R8
sm_div_loop:
	FMOVD (R7), F2
	FMULD F8, F2, F2
	FMOVD F2, (R7)
	ADD $8, R7, R7
	SUBS $1, R8, R8
	BNE sm_div_loop

sm_done:
	RET
