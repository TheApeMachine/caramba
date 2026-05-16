#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// RoPEHalfNEON(dst, src, cosTable, sinTable []float64, numPairs int)
// Rotates split-half RoPE pairs: (i, i + numPairs).
TEXT ·RoPEHalfNEON(SB), NOSPLIT, $0-104
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD cosTable+48(FP), R2
	MOVD sinTable+72(FP), R3
	MOVD numPairs+96(FP), R4

	LSL $3, R4, R5
	ADD R5, R1, R6
	ADD R5, R0, R7

vector_loop:
	CMP $2, R4
	BLT scalar_tail

	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R6), [V1.D2]
	VLD1.P 16(R2), [V2.D2]
	VLD1.P 16(R3), [V3.D2]
	VFMUL_D2(2, 0, 4)
	VFMUL_D2(3, 1, 5)
	VFSUB_D2(5, 4, 4)
	VFMUL_D2(3, 0, 5)
	VFMUL_D2(2, 1, 6)
	VFADD_D2(6, 5, 5)
	VST1.P [V4.D2], 16(R0)
	VST1.P [V5.D2], 16(R7)
	SUB $2, R4, R4
	B vector_loop

scalar_tail:
	CBZ R4, done

scalar_loop:
	FMOVD (R1), F0
	FMOVD (R6), F1
	FMOVD (R2), F2
	FMOVD (R3), F3

	FMULD F2, F0, F4
	FMULD F3, F1, F5
	FSUBD F5, F4, F6
	FMOVD F6, (R0)

	FMULD F3, F0, F7
	FMULD F2, F1, F8
	FADDD F8, F7, F9
	FMOVD F9, (R7)

	ADD $8, R1, R1
	ADD $8, R6, R6
	ADD $8, R2, R2
	ADD $8, R3, R3
	ADD $8, R0, R0
	ADD $8, R7, R7
	SUB $1, R4, R4
	CBNZ R4, scalar_loop

done:
	RET
