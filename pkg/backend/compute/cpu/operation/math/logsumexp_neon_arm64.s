#include "textflag.h"

// logSumExpRowNEON(row []float64) float64
TEXT ·logSumExpRowNEON(SB), NOSPLIT, $0-32
	MOVD row+0(FP), R0
	MOVD row_len+8(FP), R1
	CBZ R1, lsen_done_zero
	MOVD R1, R11
	MOVD R0, R12

	// Pass 1: find max
	FMOVD (R0), F10
	MOVD R0, R7
	MOVD R1, R8
lsen_max_loop:
	FMOVD (R7), F1
	FMAXD F1, F10, F10
	ADD $8, R7, R7
	SUBS $1, R8, R8
	BNE lsen_max_loop

	// Pass 2: sum of exp(row - max)
	FMOVD ·expLog2E(SB), F20
	FMOVD ·expLn2Hi(SB), F21
	FMOVD ·expLn2Lo(SB), F22
	FMOVD ·expMaxArg(SB), F23
	FMOVD ·expMinArg(SB), F24
	FMOVD $0.0, F14
	MOVD  $1023, R10

	MOVD R12, R7
	MOVD R11, R8
lsen_exp_loop:
	FMOVD (R7), F2
	FSUBD F10, F2, F2
	FMINNMD F23, F2, F2
	FMAXNMD F24, F2, F2

	FMULD F20, F2, F3
	FRINTND F3, F3
	FMSUBD F3, F2, F21, F2
	FMSUBD F3, F2, F22, F2

	FMOVD ·expC11(SB), F4
	FMOVD ·expC10(SB), F5
	FMADDD F4, F5, F2, F4
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
	FMADDD F4, F5, F2, F4

	FCVTZSD F3, R5
	ADD R10, R5, R5
	LSL $52, R5, R5
	FMOVD R5, F6
	FMULD F6, F4, F4
	FADDD F4, F14, F14

	ADD $8, R7, R7
	SUBS $1, R8, R8
	BNE lsen_exp_loop

	// Pass 3: log(sumExp) via bit decomposition + atanh polynomial
	FMOVD F14, R5
	LSR $52, R5, R6
	AND $0x7FF, R6, R6
	SUB R10, R6, R6
	SCVTFD R6, F8                               // e as double

	MOVD $0x000FFFFFFFFFFFFF, R7
	AND R7, R5, R7
	MOVD $0x3FF0000000000000, R6
	ORR R6, R7, R7
	FMOVD R7, F9                                // m in [1,2)

	FMOVD ·logSqrt2(SB), F1
	FCMPD F1, F9
	BLE lsen_no_shift
	FMOVD ·logHalf(SB), F2
	FMULD F2, F9, F9
	FMOVD $1.0, F2
	FADDD F2, F8, F8
lsen_no_shift:
	FMOVD $1.0, F1
	FSUBD F1, F9, F2                            // m-1
	FADDD F1, F9, F3                            // m+1
	FDIVD F3, F2, F2                            // t
	FMULD F2, F2, F3                            // u = t²

	FMOVD ·logA6(SB), F4
	FMOVD ·logA5(SB), F5
	FMADDD F4, F5, F3, F4
	FMOVD ·logA4(SB), F5
	FMADDD F4, F5, F3, F4
	FMOVD ·logA3(SB), F5
	FMADDD F4, F5, F3, F4
	FMOVD ·logA2(SB), F5
	FMADDD F4, F5, F3, F4
	FMOVD ·logA1(SB), F5
	FMADDD F4, F5, F3, F4
	FMOVD ·logA0(SB), F5
	FMADDD F4, F5, F3, F4                       // P(u)

	FMULD F2, F4, F4
	FADDD F4, F4, F4                            // 2*t*P = log(m)

	FMOVD ·logLn2(SB), F5
	FMADDD F5, F4, F8, F4                       // + e*ln2 = log(sumExp)
	FADDD F10, F4, F4                           // + max
	FMOVD F4, ret+24(FP)
	RET

lsen_done_zero:
	FMOVD $0.0, F0
	FMOVD F0, ret+24(FP)
	RET
