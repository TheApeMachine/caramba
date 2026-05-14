#include "textflag.h"

// bindKernelNEON(dst, a, b []float64)
TEXT ·bindKernelNEON(SB), NOSPLIT, $0-72
	MOVD a+24(FP), R1
	MOVD a_len+32(FP), R3
	MOVD b+48(FP), R2
	MOVD dst+0(FP), R0
	LSR  $1, R3, R4
	CBZ  R4, done

loop:
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R2), F2
	FMULD   F2, F0, F0
	FMOVD.P F0, 8(R0)
	FMOVD.P 8(R1), F1
	FMOVD.P 8(R2), F3
	FMULD   F3, F1, F1
	FMOVD.P F1, 8(R0)
	SUBS $1, R4, R4
	BNE  loop

done:
	RET
