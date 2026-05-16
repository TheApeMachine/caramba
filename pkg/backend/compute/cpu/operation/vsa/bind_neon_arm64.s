#include "textflag.h"

#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// bindKernelNEON(dst, a, b []float64)
TEXT ·bindKernelNEON(SB), NOSPLIT, $0-72
	MOVD a+24(FP), R1
	MOVD a_len+32(FP), R3
	MOVD b+48(FP), R2
	MOVD dst+0(FP), R0
	LSR  $1, R3, R4
	CBZ  R4, done

loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R2), [V1.D2]
	VFMUL_D2(1, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R4, R4
	BNE  loop

	TST $1, R3
	BEQ done
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R2), F2
	FMULD F2, F0, F0
	FMOVD.P F0, 8(R0)

done:
	RET
