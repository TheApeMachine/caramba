#include "textflag.h"

// bundleAccumNEON(dst, src []float64)
TEXT ·bundleAccumNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD dst_len+8(FP), R2
	CBZ R2, ba_done
ba_loop:
	FMOVD (R0), F0
	FMOVD (R1), F1
	FADDD F1, F0, F0
	FMOVD F0, (R0)
	ADD $8, R0, R0
	ADD $8, R1, R1
	SUBS $1, R2, R2
	BNE ba_loop
ba_done:
	RET

// bundleNormalizeNEON(dst []float64, eps float64)
TEXT ·bundleNormalizeNEON(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R0
	MOVD dst_len+8(FP), R1
	FMOVD eps+24(FP), F10
	MOVD R0, R3
	MOVD R1, R4

	FMOVD $0.0, F0
	CBZ R1, bn_done
bn_ss_loop:
	FMOVD (R0), F1
	FMADDD F1, F0, F1, F0
	ADD $8, R0, R0
	SUBS $1, R1, R1
	BNE bn_ss_loop

	FSQRTD F0, F0
	FCMPD F10, F0
	BLE bn_done
	FMOVD $1.0, F2
	FDIVD F0, F2, F2                            // 1/norm

	MOVD R3, R0
	MOVD R4, R1
bn_scale_loop:
	FMOVD (R0), F3
	FMULD F2, F3, F3
	FMOVD F3, (R0)
	ADD $8, R0, R0
	SUBS $1, R1, R1
	BNE bn_scale_loop

bn_done:
	RET
