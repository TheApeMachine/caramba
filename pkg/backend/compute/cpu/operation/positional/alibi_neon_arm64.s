#include "textflag.h"

// ALiBiRowNEON(dst []float64, slope float64, q int, seqLenK int)
// ABI0: dst+0, slope+24, q+32, seqLenK+40

TEXT ·ALiBiRowNEON(SB), NOSPLIT, $0-48
	MOVD  dst+0(FP),      R0
	FMOVD slope+24(FP),   F0    // slope
	MOVD  q+32(FP),       R2
	MOVD  seqLenK+40(FP), R3

	SCVTFD R2, F1               // float64(q)
	FMULD  F0, F1, F1           // slope*q

	MOVD  $0, R4                // k = 0
loop:
	CBZ R3, done

	SCVTFD R4, F2               // float64(k)
	FMULD  F0, F2, F2           // slope*k
	FSUBD  F1, F2, F2           // slope*k - slope*q
	FMOVD  F2, (R0)

	ADD $8, R0
	ADD $1, R4
	SUB $1, R3
	B   loop

done:
	RET
