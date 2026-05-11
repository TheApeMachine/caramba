#include "textflag.h"

// RoPENEON(dst, src, cosTable, sinTable []float64, numPairs int)
// ABI0: dst+0, src+24, cosTable+48, sinTable+72, numPairs+96
// Processes one pair per iteration using ARM64 scalar float64.

TEXT ·RoPENEON(SB), NOSPLIT, $0-104
	MOVD dst+0(FP),      R0
	MOVD src+24(FP),     R1
	MOVD cosTable+48(FP), R2
	MOVD sinTable+72(FP), R3
	MOVD numPairs+96(FP), R4

loop:
	CBZ   R4, done

	FMOVD (R1), F0      // x[2i]
	FMOVD 8(R1), F1     // x[2i+1]
	FMOVD (R2), F2      // cos[i]
	FMOVD (R3), F3      // sin[i]

	// d[2i] = x0*cos - x1*sin
	FMULD F2, F0, F4    // x0*cos
	FMULD F3, F1, F5    // x1*sin
	FSUBD F5, F4, F6    // x0*cos - x1*sin
	FMOVD F6, (R0)

	// d[2i+1] = x0*sin + x1*cos
	FMULD F3, F0, F7    // x0*sin
	FMULD F2, F1, F8    // x1*cos
	FADDD F8, F7, F9    // x0*sin + x1*cos
	FMOVD F9, 8(R0)

	ADD $16, R1
	ADD $16, R0
	ADD $8,  R2
	ADD $8,  R3
	SUB $1,  R4
	B   loop

done:
	RET
