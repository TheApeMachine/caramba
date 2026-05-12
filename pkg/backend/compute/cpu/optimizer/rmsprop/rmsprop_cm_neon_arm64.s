#include "textflag.h"

// rmspropCenteredMomentumNEON(out, v, gAvg, buf, params, grads []float64,
//                             lr, alpha, oneMinusAlpha, eps, momentum, wd float64)
TEXT ·rmspropCenteredMomentumNEON(SB), NOSPLIT, $0-192
	MOVD out+0(FP), R0
	MOVD v+24(FP), R1
	MOVD gAvg+48(FP), R7
	MOVD buf+72(FP), R8
	MOVD params+96(FP), R2
	MOVD grads+120(FP), R3
	MOVD out_len+8(FP), R4

	FMOVD lr+144(FP), F20
	FMOVD alpha+152(FP), F21
	FMOVD oneMinusAlpha+160(FP), F22
	FMOVD eps+168(FP), F23
	FMOVD momentum+176(FP), F24
	FMOVD wd+184(FP), F25

	CBZ R4, rmscm_neon_done
rmscm_neon_loop:
	FMOVD (R1), F0
	FMOVD (R7), F1
	FMOVD (R8), F14
	FMOVD (R2), F2
	FMOVD (R3), F3

	FMADDD F25, F2, F3, F4                     // geff
	FMULD F21, F0, F0
	FMULD F4, F4, F5
	FMADDD F5, F0, F22, F0
	FMOVD F0, (R1)

	FMULD F21, F1, F1
	FMADDD F4, F1, F22, F1
	FMOVD F1, (R7)

	FMULD F1, F1, F6
	FSUBD F6, F0, F6
	FSQRTD F6, F6
	FADDD F23, F6, F6
	FDIVD F6, F4, F7

	FMULD F24, F14, F14
	FMADDD F7, F14, F20, F14
	FMOVD F14, (R8)

	FSUBD F14, F2, F2
	FMOVD F2, (R0)

	ADD $8, R0, R0
	ADD $8, R1, R1
	ADD $8, R7, R7
	ADD $8, R8, R8
	ADD $8, R2, R2
	ADD $8, R3, R3
	SUBS $1, R4, R4
	BNE rmscm_neon_loop

rmscm_neon_done:
	RET
