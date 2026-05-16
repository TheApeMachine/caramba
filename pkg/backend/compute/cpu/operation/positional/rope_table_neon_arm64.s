#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// ropeAdvanceRowNEON(cosCur, sinCur, cosStep, sinStep []float64)
TEXT ·ropeAdvanceRowNEON(SB), NOSPLIT, $0-96
	MOVD cosCur+0(FP), R0
	MOVD sinCur+24(FP), R1
	MOVD cosStep+48(FP), R2
	MOVD sinStep+72(FP), R3
	MOVD cosCur_len+8(FP), R4
	CBZ R4, ra_done
	LSR $1, R4, R5
	CBZ R5, ra_tail
ra_loop:
	VLD1.P 16(R0), [V0.D2]
	VLD1.P 16(R1), [V1.D2]
	VLD1.P 16(R2), [V2.D2]
	VLD1.P 16(R3), [V3.D2]
	VFMUL_D2(2, 0, 4)
	VFMUL_D2(3, 1, 5)
	VFSUB_D2(5, 4, 4)
	VFMUL_D2(2, 1, 5)
	VFMUL_D2(3, 0, 6)
	VFADD_D2(6, 5, 5)
	SUB $16, R0, R0
	SUB $16, R1, R1
	VST1.P [V4.D2], 16(R0)
	VST1.P [V5.D2], 16(R1)
	SUBS $1, R5, R5
	BNE ra_loop

ra_tail:
	TST $1, R4
	BEQ ra_done
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
ra_done:
	RET
