#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// RoPENEON(dst, src, cosTable, sinTable []float64, numPairs int)
// ABI0: dst+0, src+24, cosTable+48, sinTable+72, numPairs+96
// Processes two rotary pairs per iteration using ARM64 NEON.

TEXT ·RoPENEON(SB), NOSPLIT, $0-104
	MOVD dst+0(FP),      R0
	MOVD src+24(FP),     R1
	MOVD cosTable+48(FP), R2
	MOVD sinTable+72(FP), R3
	MOVD numPairs+96(FP), R4

vector_loop:
	CMP $2, R4
	BLT scalar_tail

	VLD2.P 32(R1), [V0.D2, V1.D2]       // even x, odd x
	VLD1.P 16(R2), [V2.D2]              // cos
	VLD1.P 16(R3), [V3.D2]              // sin
	VFMUL_D2(2, 0, 4)                   // even*cos
	VFMUL_D2(3, 1, 5)                   // odd*sin
	VFSUB_D2(5, 4, 4)
	VFMUL_D2(3, 0, 5)                   // even*sin
	VFMUL_D2(2, 1, 6)                   // odd*cos
	VFADD_D2(6, 5, 5)
	VST2.P [V4.D2, V5.D2], 32(R0)
	SUB $2, R4, R4
	B vector_loop

scalar_tail:
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
	B   scalar_tail

done:
	RET
