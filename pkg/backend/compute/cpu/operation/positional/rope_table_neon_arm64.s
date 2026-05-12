#include "textflag.h"

// ropeAdvanceRowNEON(cosCur, sinCur, cosStep, sinStep []float64)
TEXT ·ropeAdvanceRowNEON(SB), NOSPLIT, $0-96
	MOVD cosCur+0(FP), R0
	MOVD sinCur+24(FP), R1
	MOVD cosStep+48(FP), R2
	MOVD sinStep+72(FP), R3
	MOVD cosCur_len+8(FP), R4
	CBZ R4, ra_done
ra_loop:
	FMOVD (R0), F0                            // cosCur
	FMOVD (R1), F1                            // sinCur
	FMOVD (R2), F2                            // cosStep
	FMOVD (R3), F3                            // sinStep
	// cosNext = cosCur*cosStep - sinCur*sinStep
	FMULD F2, F0, F4                          // cosCur*cosStep
	FMSUBD F3, F4, F1, F4                     // - sinCur*sinStep   (FMSUBD A,B,C,D: D=B-A*C → F4=F4-F3*F1)
	// sinNext = sinCur*cosStep + cosCur*sinStep
	FMULD F2, F1, F5                          // sinCur*cosStep
	FMADDD F3, F5, F0, F5                     // + cosCur*sinStep
	FMOVD F4, (R0)
	FMOVD F5, (R1)
	ADD $8, R0, R0
	ADD $8, R1, R1
	ADD $8, R2, R2
	ADD $8, R3, R3
	SUBS $1, R4, R4
	BNE ra_loop
ra_done:
	RET
