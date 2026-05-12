#include "textflag.h"

// rmspropPlainNEON(out, v, params, grads []float64,
//                  lr, alpha, oneMinusAlpha, eps, wd float64)
TEXT ·rmspropPlainNEON(SB), NOSPLIT, $0-136
	MOVD out+0(FP), R0
	MOVD v+24(FP), R1
	MOVD params+48(FP), R2
	MOVD grads+72(FP), R3
	MOVD out_len+8(FP), R4

	FMOVD lr+96(FP), F20
	FMOVD alpha+104(FP), F21
	FMOVD oneMinusAlpha+112(FP), F22
	FMOVD eps+120(FP), F23
	FMOVD wd+128(FP), F24

	LSR $1, R4, R5
	CBZ R5, rmsp_neon_tail
rmsp_neon_loop:
	FMOVD (R1), F0
	FMOVD 8(R1), F1
	FMOVD (R2), F2
	FMOVD 8(R2), F3
	FMOVD (R3), F4
	FMOVD 8(R3), F5

	// geff = g + wd*p
	FMADDD F24, F4, F2, F6
	FMADDD F24, F5, F3, F7

	// v = α*v + (1-α)*geff²
	FMULD F21, F0, F0
	FMULD F21, F1, F1
	FMULD F6, F6, F8
	FMULD F7, F7, F9
	FMADDD F8, F0, F22, F0
	FMADDD F9, F1, F22, F1
	FMOVD F0, (R1)
	FMOVD F1, 8(R1)

	// denom = sqrt(v) + eps
	FSQRTD F0, F10
	FSQRTD F1, F11
	FADDD F23, F10, F10
	FADDD F23, F11, F11
	// upd = geff / denom
	FDIVD F10, F6, F12
	FDIVD F11, F7, F13
	// out = p - lr*upd
	FMSUBD F20, F2, F12, F2
	FMSUBD F20, F3, F13, F3
	FMOVD F2, (R0)
	FMOVD F3, 8(R0)

	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	ADD $16, R3, R3
	SUBS $1, R5, R5
	BNE  rmsp_neon_loop

rmsp_neon_tail:
	AND $1, R4, R6
	CBZ R6, rmsp_neon_done
	FMOVD (R1), F0
	FMOVD (R2), F2
	FMOVD (R3), F4
	FMADDD F24, F4, F2, F6
	FMULD F21, F0, F0
	FMULD F6, F6, F8
	FMADDD F8, F0, F22, F0
	FMOVD F0, (R1)
	FSQRTD F0, F10
	FADDD F23, F10, F10
	FDIVD F10, F6, F12
	FMSUBD F20, F2, F12, F2
	FMOVD F2, (R0)

rmsp_neon_done:
	RET
