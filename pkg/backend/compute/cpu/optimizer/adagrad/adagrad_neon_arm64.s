#include "textflag.h"

// adagradStepNEON(out, G, params, grads []float64, lr, eps, wd float64)
TEXT ·adagradStepNEON(SB), NOSPLIT, $0-120
	MOVD out+0(FP), R0
	MOVD G+24(FP), R1
	MOVD params+48(FP), R2
	MOVD grads+72(FP), R3
	MOVD out_len+8(FP), R4

	FMOVD lr+96(FP), F20
	FMOVD eps+104(FP), F21
	FMOVD wd+112(FP), F22

	LSR  $1, R4, R5
	CBZ  R5, ag_neon_tail
ag_neon_loop:
	FMOVD (R1), F0
	FMOVD 8(R1), F1
	FMOVD (R2), F2
	FMOVD 8(R2), F3
	FMOVD (R3), F4
	FMOVD 8(R3), F5

	// geff = grads + wd*params  (FMADDD A, B, C, D : D = A*C + B)
	FMADDD F22, F4, F2, F6
	FMADDD F22, F5, F3, F7

	// G += geff^2  via FMADDD: G_new = G_old + geff*geff
	FMADDD F6, F0, F6, F0
	FMADDD F7, F1, F7, F1
	FMOVD F0, (R1)
	FMOVD F1, 8(R1)

	// denom = sqrt(G) + eps
	FSQRTD F0, F8
	FSQRTD F1, F9
	FADDD F21, F8, F8
	FADDD F21, F9, F9

	// upd = geff / denom
	FDIVD F8, F6, F10
	FDIVD F9, F7, F11

	// out = params - lr*upd  (FMSUBD A, B, C, D : D = B - A*C)
	FMSUBD F20, F2, F10, F2
	FMSUBD F20, F3, F11, F3
	FMOVD F2, (R0)
	FMOVD F3, 8(R0)

	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	ADD $16, R3, R3
	SUBS $1, R5, R5
	BNE  ag_neon_loop

ag_neon_tail:
	AND $1, R4, R6
	CBZ R6, ag_neon_done
	FMOVD (R1), F0
	FMOVD (R2), F2
	FMOVD (R3), F4
	FMADDD F22, F4, F2, F6
	FMADDD F6, F0, F6, F0
	FMOVD F0, (R1)
	FSQRTD F0, F8
	FADDD F21, F8, F8
	FDIVD F8, F6, F10
	FMSUBD F20, F2, F10, F2
	FMOVD F2, (R0)

ag_neon_done:
	RET

// adadeltaStepNEON(out, eg2, edp2, params, grads []float64, rho, eps, wd float64)
//   geff = grads + wd*params
//   eg2 = ρ*eg2 + (1-ρ)*geff²
//   numer = sqrt(edp2 + eps)
//   denom = sqrt(eg2 + eps)
//   delta = -(numer/denom)*geff
//   edp2 = ρ*edp2 + (1-ρ)*delta²
//   out = params + delta
TEXT ·adadeltaStepNEON(SB), NOSPLIT, $0-152
	MOVD out+0(FP), R0
	MOVD eg2+24(FP), R1
	MOVD edp2+48(FP), R2
	MOVD params+72(FP), R3
	MOVD grads+96(FP), R4
	MOVD out_len+8(FP), R5

	FMOVD rho+120(FP), F20
	FMOVD oneMinusRho+128(FP), F21
	FMOVD eps+136(FP), F22
	FMOVD wd+144(FP), F23

	LSR  $1, R5, R6
	CBZ  R6, ad_neon_tail
ad_neon_loop:
	FMOVD (R1), F0
	FMOVD 8(R1), F1
	FMOVD (R2), F2
	FMOVD 8(R2), F3
	FMOVD (R3), F4
	FMOVD 8(R3), F5
	FMOVD (R4), F6
	FMOVD 8(R4), F7

	// geff = grads + wd*params
	FMADDD F23, F4, F6, F6
	FMADDD F23, F5, F7, F7

	// eg2 = ρ*eg2 + (1-ρ)*geff²
	FMULD F20, F0, F0
	FMULD F20, F1, F1
	FMULD F6, F6, F8
	FMULD F7, F7, F9
	FMADDD F8, F0, F21, F0
	FMADDD F9, F1, F21, F1
	FMOVD F0, (R1)
	FMOVD F1, 8(R1)

	// numer = sqrt(edp2 + eps)
	FADDD F22, F2, F10
	FADDD F22, F3, F11
	FSQRTD F10, F10
	FSQRTD F11, F11
	// denom = sqrt(eg2 + eps)
	FADDD F22, F0, F12
	FADDD F22, F1, F13
	FSQRTD F12, F12
	FSQRTD F13, F13

	// delta = -(numer/denom)*geff
	FDIVD F12, F10, F14
	FDIVD F13, F11, F15
	FMULD F6, F14, F14
	FMULD F7, F15, F15
	FNEGD F14, F14
	FNEGD F15, F15

	// edp2 = ρ*edp2 + (1-ρ)*delta²
	FMULD F20, F2, F2
	FMULD F20, F3, F3
	FMULD F14, F14, F16
	FMULD F15, F15, F17
	FMADDD F16, F2, F21, F2
	FMADDD F17, F3, F21, F3
	FMOVD F2, (R2)
	FMOVD F3, 8(R2)

	// out = params + delta
	FADDD F14, F4, F4
	FADDD F15, F5, F5
	FMOVD F4, (R0)
	FMOVD F5, 8(R0)

	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	ADD $16, R3, R3
	ADD $16, R4, R4
	SUBS $1, R6, R6
	BNE  ad_neon_loop

ad_neon_tail:
	AND $1, R5, R7
	CBZ R7, ad_neon_done
	FMOVD (R1), F0
	FMOVD (R2), F2
	FMOVD (R3), F4
	FMOVD (R4), F6
	FMADDD F23, F4, F6, F6
	FMULD F20, F0, F0
	FMULD F6, F6, F8
	FMADDD F8, F0, F21, F0
	FMOVD F0, (R1)
	FADDD F22, F2, F10
	FSQRTD F10, F10
	FADDD F22, F0, F12
	FSQRTD F12, F12
	FDIVD F12, F10, F14
	FMULD F6, F14, F14
	FNEGD F14, F14
	FMULD F20, F2, F2
	FMULD F14, F14, F16
	FMADDD F16, F2, F21, F2
	FMOVD F2, (R2)
	FADDD F14, F4, F4
	FMOVD F4, (R0)

ad_neon_done:
	RET
