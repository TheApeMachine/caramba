#include "textflag.h"

// rmsNormRowNEON(out, row, weight []float64, eps float64)
TEXT ·rmsNormRowNEON(SB), NOSPLIT, $0-80
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
rms_sum_loop:
	FMOVD (R6), F1
	FMADDD F1, F0, F1, F0
	ADD $8, R6, R6
	SUBS $1, R7, R7
	BNE rms_sum_loop
rms_post_sum:
	SCVTFD R4, F4
	FDIVD F4, F0, F0
	FMOVD eps+72(FP), F5
	FADDD F5, F0, F0
	FSQRTD F0, F0
	FMOVD $1.0, F2
	FDIVD F0, F2, F2

	MOVD R5, R6
	MOVD R3, R7
	CBZ R7, rms_done
rms_norm_loop:
	FMOVD (R6), F1
	FMULD F2, F1, F1
	FMOVD (R2), F3
	FMULD F3, F1, F1
	FMOVD F1, (R0)
	ADD $8, R0, R0
	ADD $8, R6, R6
	ADD $8, R2, R2
	SUBS $1, R7, R7
	BNE rms_norm_loop
rms_done:
	RET
