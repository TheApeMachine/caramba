#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// rmsNormRowNEON(out, row, weight []float64, eps float64)
TEXT ·rmsNormRowNEON(SB), NOSPLIT, $24-80
	MOVD out+0(FP), R0
	MOVD row+24(FP), R1
	MOVD weight+48(FP), R2
	MOVD out_len+8(FP), R3
	MOVD R3, R4
	MOVD R1, R5
	MOVD R1, R6
	FMOVD $0.0, F0
	MOVD R3, R7

	CBZ R7, rms_post_sum
	VEOR V0.B16, V0.B16, V0.B16
	LSR  $1, R7, R8
	CBZ  R8, rms_sum_tail
rms_sum_loop:
	VLD1.P 16(R6), [V1.D2]
	VFMUL_D2(1, 1, 2)
	VFADD_D2(2, 0, 0)
	SUBS $1, R8, R8
	BNE  rms_sum_loop

	MOVD RSP, R8
	VST1.P [V0.D2], 16(R8)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0

rms_sum_tail:
	TST $1, R7
	BEQ rms_post_sum
	FMOVD (R6), F1
	FMADDD F1, F0, F1, F0
rms_post_sum:
	SCVTFD R4, F4
	FDIVD F4, F0, F0
	FMOVD eps+72(FP), F5
	FADDD F5, F0, F0
	FSQRTD F0, F0
	FMOVD $1.0, F2
	FDIVD F0, F2, F2
	FMOVD F2, 16(RSP)
	ADD $16, RSP, R8
	VLD1R (R8), [V20.D2]

	MOVD R5, R6
	MOVD R3, R7
	CBZ R7, rms_done
	LSR $1, R7, R8
	CBZ R8, rms_norm_tail
rms_norm_loop:
	VLD1.P 16(R6), [V1.D2]
	VLD1.P 16(R2), [V2.D2]
	VFMUL_D2(20, 1, 1)
	VFMUL_D2(2, 1, 1)
	VST1.P [V1.D2], 16(R0)
	SUBS $1, R8, R8
	BNE  rms_norm_loop

rms_norm_tail:
	TST $1, R7
	BEQ rms_done
	FMOVD (R6), F1
	FMULD F2, F1, F1
	FMOVD (R2), F3
	FMULD F3, F1, F1
	FMOVD F1, (R0)
rms_done:
	RET
