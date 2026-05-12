#include "textflag.h"

// hawkesKernelRowNEON(out, events []float64, ti, alpha, beta float64)
TEXT ·hawkesKernelRowNEON(SB), NOSPLIT, $0-72
	MOVD out+0(FP), R0
	MOVD events+24(FP), R1
	MOVD out_len+8(FP), R2
	CBZ R2, hkr_done

	FMOVD ti+48(FP), F20
	FMOVD beta+64(FP), F21
	FMOVD alpha+56(FP), F22
	FMOVD ·hexLog2E(SB), F23
	FMOVD ·hexLn2Hi(SB), F24
	FMOVD ·hexLn2Lo(SB), F25
	FMOVD ·hexMaxArg(SB), F26
	FMOVD ·hexMinArg(SB), F27
	MOVD  $1023, R10

hkr_loop:
	FMOVD (R1), F0
	FSUBD F20, F0, F0                          // events - ti
	FMULD F21, F0, F0                          // *beta
	FNEGD F0, F0                               // -beta*(events-ti)
	FMINNMD F26, F0, F0
	FMAXNMD F27, F0, F0

	FMULD F23, F0, F1
	FRINTND F1, F1
	FMSUBD F1, F0, F24, F0
	FMSUBD F1, F0, F25, F0

	FMOVD ·hexC11(SB), F2
	FMOVD ·hexC10(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·hexC9(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·hexC8(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·hexC7(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·hexC6(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·hexC5(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·hexC4(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·hexC3(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·hexC2(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·hexC1(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·hexC0(SB), F3
	FMADDD F2, F3, F0, F2

	FCVTZSD F1, R5
	ADD R10, R5, R5
	LSL $52, R5, R5
	FMOVD R5, F4
	FMULD F4, F2, F2
	FMULD F22, F2, F2
	FMOVD F2, (R0)

	ADD $8, R0, R0
	ADD $8, R1, R1
	SUBS $1, R2, R2
	BNE hkr_loop

hkr_done:
	RET
