#include "textflag.h"

// similarityKernelNEON(a, b []float64) float64
TEXT ·similarityKernelNEON(SB), NOSPLIT, $0-56
	MOVD  a+0(FP), R0
	MOVD  a_len+8(FP), R1
	MOVD  b+24(FP), R2
	FMOVD $0.0, F0
	FMOVD $0.0, F5
	LSR   $2, R1, R3
	CBZ   R3, try_pair

loop_quad:
	FMOVD.P 8(R0), F1
	FMOVD.P 8(R2), F3
	FMULD   F3, F1, F1
	FADDD   F1, F0, F0
	FMOVD.P 8(R0), F2
	FMOVD.P 8(R2), F4
	FMULD   F4, F2, F2
	FADDD   F2, F5, F5
	FMOVD.P 8(R0), F6
	FMOVD.P 8(R2), F7
	FMULD   F7, F6, F6
	FADDD   F6, F0, F0
	FMOVD.P 8(R0), F8
	FMOVD.P 8(R2), F9
	FMULD   F9, F8, F8
	FADDD   F8, F5, F5
	SUBS $1, R3, R3
	BNE  loop_quad

try_pair:
	AND $3, R1, R1
	LSR $1, R1, R3
	CBZ R3, tail

loop_pair:
	FMOVD.P 8(R0), F1
	FMOVD.P 8(R2), F3
	FMULD   F3, F1, F1
	FADDD   F1, F0, F0
	FMOVD.P 8(R0), F2
	FMOVD.P 8(R2), F4
	FMULD   F4, F2, F2
	FADDD   F2, F5, F5
	SUBS $1, R3, R3
	BNE  loop_pair

tail:
	FADDD F5, F0, F0
	FMOVD F0, ret+48(FP)
	RET
