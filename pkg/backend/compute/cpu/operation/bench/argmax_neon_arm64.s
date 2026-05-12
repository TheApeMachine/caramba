#include "textflag.h"

// argmaxNEON(xs []float64) int
TEXT ·argmaxNEON(SB), NOSPLIT, $0-32
	MOVD xs+0(FP), R0
	MOVD xs_len+8(FP), R1
	MOVD $0, R2                                // best
	CBZ R1, am_done
	FMOVD (R0), F0                             // best value
	MOVD $0, R3                                // i
am_loop:
	ADD $1, R3, R3
	CMP R1, R3
	BGE am_done
	LSL $3, R3, R4
	ADD R0, R4, R4
	FMOVD (R4), F1
	FCMPD F0, F1
	BLE am_loop
	FMOVD F1, F0
	MOVD R3, R2
	B am_loop
am_done:
	MOVD R2, ret+24(FP)
	RET
