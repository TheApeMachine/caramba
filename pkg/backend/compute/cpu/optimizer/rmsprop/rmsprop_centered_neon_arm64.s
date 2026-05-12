#include "textflag.h"

// rmspropCenteredNEON(out, v, gAvg, params, grads []float64,
//                     lr, alpha, oneMinusAlpha, eps, wd float64)
TEXT ·rmspropCenteredNEON(SB), NOSPLIT, $0-160
	MOVD out+0(FP), R0
	MOVD v+24(FP), R1
	MOVD gAvg+48(FP), R7
	MOVD params+72(FP), R2
	MOVD grads+96(FP), R3
	MOVD out_len+8(FP), R4

	FMOVD lr+120(FP), F20
	FMOVD alpha+128(FP), F21
	FMOVD oneMinusAlpha+136(FP), F22
	FMOVD eps+144(FP), F23
	FMOVD wd+152(FP), F24

	CBZ R4, rmspc_neon_done
rmspc_neon_loop:
	FMOVD (R1), F0
	FMOVD (R7), F1
	FMOVD (R2), F2
	FMOVD (R3), F3

	FMADDD F24, F2, F3, F4                     // geff
	FMULD F21, F0, F0
	FMULD F4, F4, F5
	FMADDD F5, F0, F22, F0                     // v
	FMOVD F0, (R1)

	FMULD F21, F1, F1
	FMADDD F4, F1, F22, F1                     // gAvg
	FMOVD F1, (R7)

	FMULD F1, F1, F6
	FSUBD F6, F0, F6
	FSQRTD F6, F6
	FADDD F23, F6, F6
	FDIVD F6, F4, F7
	FMSUBD F20, F2, F7, F2
	FMOVD F2, (R0)

	ADD $8, R0, R0
	ADD $8, R1, R1
	ADD $8, R7, R7
	ADD $8, R2, R2
	ADD $8, R3, R3
	SUBS $1, R4, R4
	BNE rmspc_neon_loop

rmspc_neon_done:
	RET

// rmspropMomentumNEON(out, v, buf, params, grads []float64,
//                     lr, alpha, oneMinusAlpha, eps, momentum, wd float64)
TEXT ·rmspropMomentumNEON(SB), NOSPLIT, $0-168
	MOVD out+0(FP), R0
	MOVD v+24(FP), R1
	MOVD buf+48(FP), R7
	MOVD params+72(FP), R2
	MOVD grads+96(FP), R3
	MOVD out_len+8(FP), R4

	FMOVD lr+120(FP), F20
	FMOVD alpha+128(FP), F21
	FMOVD oneMinusAlpha+136(FP), F22
	FMOVD eps+144(FP), F23
	FMOVD momentum+152(FP), F24
	FMOVD wd+160(FP), F25

	CBZ R4, rmspm_neon_done
rmspm_neon_loop:
	FMOVD (R1), F0
	FMOVD (R7), F1
	FMOVD (R2), F2
	FMOVD (R3), F3

	FMADDD F25, F2, F3, F4                     // geff
	FMULD F21, F0, F0
	FMULD F4, F4, F5
	FMADDD F5, F0, F22, F0
	FMOVD F0, (R1)

	FSQRTD F0, F6
	FADDD F23, F6, F6
	FDIVD F6, F4, F7

	FMULD F24, F1, F1
	FMADDD F7, F1, F20, F1                     // buf = μ*buf + lr*upd
	FMOVD F1, (R7)

	FSUBD F1, F2, F2
	FMOVD F2, (R0)

	ADD $8, R0, R0
	ADD $8, R1, R1
	ADD $8, R7, R7
	ADD $8, R2, R2
	ADD $8, R3, R3
	SUBS $1, R4, R4
	BNE rmspm_neon_loop

rmspm_neon_done:
	RET
