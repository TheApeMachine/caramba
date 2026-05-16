#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// ALiBiRowNEON(dst []float64, slope float64, q int, seqLenK int)
// ABI0: dst+0, slope+24, q+32, seqLenK+40

TEXT ·ALiBiRowNEON(SB), NOSPLIT, $16-48
	MOVD  dst+0(FP),      R0
	FMOVD slope+24(FP),   F0    // slope
	MOVD  q+32(FP),       R2
	MOVD  seqLenK+40(FP), R3

	SCVTFD R2, F1               // float64(q)
	FMULD  F0, F1, F1           // slope*q
	FNEGD  F1, F1               // -slope*q

	FMOVD F0, 0(RSP)
	VLD1R (RSP), [V0.D2]        // slope
	FMOVD F1, 0(RSP)
	VLD1R (RSP), [V1.D2]        // base

	MOVD   $0, R4
	MOVD   $1, R5
	SCVTFD R4, F2
	SCVTFD R5, F3
	FMOVD  F2, 0(RSP)
	FMOVD  F3, 8(RSP)
	MOVD   RSP, R6
	VLD1   (R6), [V2.D2]        // k offsets

	MOVD   $2, R5
	SCVTFD R5, F3
	FMOVD  F3, 0(RSP)
	VLD1R  (RSP), [V3.D2]       // +2

loop:
	CMP  $2, R3
	BLT  tail
	VFMUL_D2(0, 2, 4)
	VFADD_D2(1, 4, 4)
	VST1.P [V4.D2], 16(R0)
	VFADD_D2(3, 2, 2)
	ADD  $2, R4, R4
	SUB  $2, R3, R3
	B    loop

tail:
	CBZ R3, done

	SCVTFD R4, F2
	FMULD F0, F2, F2
	FADDD F1, F2, F2
	FMOVD F2, (R0)

done:
	RET
