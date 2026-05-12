#include "textflag.h"

// sgdVanillaNEON(out, params, grads []float64, lr, wd float64)
TEXT ·sgdVanillaNEON(SB), NOSPLIT, $0-104
	MOVD out+0(FP), R0
	MOVD params+24(FP), R1
	MOVD grads+48(FP), R2
	MOVD out_len+8(FP), R3
	FMOVD lr+72(FP), F20
	FMOVD wd+80(FP), F21

	LSR  $1, R3, R4
	CBZ  R4, sgdv_neon_tail
sgdv_neon_loop:
	FMOVD (R1), F0
	FMOVD 8(R1), F1
	FMOVD (R2), F2
	FMOVD 8(R2), F3

	// geff = g + wd*p  (FMADDD A, B, C, D : D = A*C + B)
	FMADDD F21, F2, F0, F2
	FMADDD F21, F3, F1, F3
	// out = p - lr*geff  (FMSUBD A, B, C, D : D = B - A*C)
	FMSUBD F20, F0, F2, F0
	FMSUBD F20, F1, F3, F1
	FMOVD F0, (R0)
	FMOVD F1, 8(R0)

	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	SUBS $1, R4, R4
	BNE  sgdv_neon_loop

sgdv_neon_tail:
	AND $1, R3, R5
	CBZ R5, sgdv_neon_done
	FMOVD (R1), F0
	FMOVD (R2), F2
	FMADDD F21, F2, F0, F2
	FMSUBD F20, F0, F2, F0
	FMOVD F0, (R0)
sgdv_neon_done:
	RET

// sgdMomentumNEON(out, params, grads, velocity []float64,
//                 lr, wd, momentum float64, nesterov uint64)
TEXT ·sgdMomentumNEON(SB), NOSPLIT, $0-136
	MOVD out+0(FP), R0
	MOVD params+24(FP), R1
	MOVD grads+48(FP), R2
	MOVD velocity+72(FP), R3
	MOVD out_len+8(FP), R4
	FMOVD lr+96(FP), F20
	FMOVD wd+104(FP), F21
	FMOVD momentum+112(FP), F22
	MOVD nesterov+120(FP), R12

	LSR  $1, R4, R5
	CBZ  R5, sgdm_neon_tail
sgdm_neon_loop:
	FMOVD (R1), F0
	FMOVD 8(R1), F1
	FMOVD (R2), F2
	FMOVD 8(R2), F3
	FMOVD (R3), F4
	FMOVD 8(R3), F5

	// v = μ*v - lr*g
	FMULD F22, F4, F4
	FMULD F22, F5, F5
	FMSUBD F20, F4, F2, F4
	FMSUBD F20, F5, F3, F5
	FMOVD F4, (R3)
	FMOVD F5, 8(R3)

	// out = p - lr*wd*p  ≡  p*(1 - lr*wd)
	FMULD F21, F0, F6
	FMULD F21, F1, F7
	FMSUBD F20, F0, F6, F6
	FMSUBD F20, F1, F7, F7

	// add velocity contribution
	CBZ R12, sgdm_neon_addV
	// Nesterov: out += μ*v - lr*g
	FMULD F22, F4, F8
	FMULD F22, F5, F9
	FMSUBD F20, F8, F2, F8
	FMSUBD F20, F9, F3, F9
	FADDD F8, F6, F6
	FADDD F9, F7, F7
	B sgdm_neon_store
sgdm_neon_addV:
	FADDD F4, F6, F6
	FADDD F5, F7, F7
sgdm_neon_store:
	FMOVD F6, (R0)
	FMOVD F7, 8(R0)

	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	ADD $16, R3, R3
	SUBS $1, R5, R5
	BNE  sgdm_neon_loop

sgdm_neon_tail:
	AND $1, R4, R6
	CBZ R6, sgdm_neon_done
	FMOVD (R1), F0
	FMOVD (R2), F2
	FMOVD (R3), F4

	FMULD F22, F4, F4
	FMSUBD F20, F4, F2, F4
	FMOVD F4, (R3)

	FMULD F21, F0, F6
	FMSUBD F20, F0, F6, F6
	CBZ R12, sgdm_neon_addV2
	FMULD F22, F4, F8
	FMSUBD F20, F8, F2, F8
	FADDD F8, F6, F6
	B sgdm_neon_storeTail
sgdm_neon_addV2:
	FADDD F4, F6, F6
sgdm_neon_storeTail:
	FMOVD F6, (R0)

sgdm_neon_done:
	RET
