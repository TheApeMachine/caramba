#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// similarityKernelNEON(a, b []float64) float64
TEXT ·similarityKernelNEON(SB), NOSPLIT, $16-56
	MOVD  a+0(FP), R0
	MOVD  a_len+8(FP), R1
	MOVD  b+24(FP), R2
	FMOVD $0.0, F0
	VEOR V0.B16, V0.B16, V0.B16
	LSR $1, R1, R3
	CBZ R3, tail

loop_pair:
	VLD1.P 16(R0), [V1.D2]
	VLD1.P 16(R2), [V2.D2]
	VFMUL_D2(2, 1, 1)
	VFADD_D2(1, 0, 0)
	SUBS $1, R3, R3
	BNE  loop_pair

	MOVD RSP, R3
	VST1.P [V0.D2], 16(R3)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0

tail:
	TST $1, R1
	BEQ done
	FMOVD.P 8(R0), F1
	FMOVD.P 8(R2), F2
	FMULD F2, F1, F1
	FADDD F1, F0, F0

done:
	FMOVD F0, ret+48(FP)
	RET
