#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))

// expSumNEON(expBuf []float64) float64
TEXT ·expSumNEON(SB), NOSPLIT, $16-32
	MOVD expBuf+0(FP), R0
	MOVD expBuf_len+8(FP), R1
	VEOR V0.B16, V0.B16, V0.B16
	LSR  $1, R1, R2
	CBZ  R2, exp_sum_neon_tail
exp_sum_neon_loop:
	VLD1.P 16(R0), [V1.D2]
	VFADD_D2(1, 0, 0)
	SUBS $1, R2, R2
	BNE  exp_sum_neon_loop
	MOVD RSP, R2
	VST1.P [V0.D2], 16(R2)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0
exp_sum_neon_tail:
	TST $1, R1
	BEQ exp_sum_neon_done
	FMOVD.P 8(R0), F1
	FADDD F1, F0, F0
exp_sum_neon_done:
	FMOVD F0, ret+24(FP)
	RET
