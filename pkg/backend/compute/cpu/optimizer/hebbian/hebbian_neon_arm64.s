#include "textflag.h"

// hebbStepNEON(out, params, grads []float64, lr float64)
TEXT ·hebbStepNEON(SB), NOSPLIT, $0-80
	MOVD out+0(FP), R0
	MOVD params+24(FP), R1
	MOVD grads+48(FP), R2
	MOVD out_len+8(FP), R3
	FMOVD lr+72(FP), F20
	CBZ R3, hebb_neon_done
hebb_neon_loop:
	FMOVD (R1), F0
	FMOVD (R2), F1
	FMADDD F20, F0, F1, F0
	FMOVD F0, (R0)
	ADD $8, R0, R0
	ADD $8, R1, R1
	ADD $8, R2, R2
	SUBS $1, R3, R3
	BNE hebb_neon_loop
hebb_neon_done:
	RET

// hebbStepNormNEON(out, params, grads []float64, lr float64) float64
// returns L2 norm of updated weights.
TEXT ·hebbStepNormNEON(SB), NOSPLIT, $0-88
	MOVD out+0(FP), R0
	MOVD params+24(FP), R1
	MOVD grads+48(FP), R2
	MOVD out_len+8(FP), R3
	FMOVD lr+72(FP), F20
	FMOVD $0.0, F10
	CBZ R3, hebbn_neon_done
hebbn_neon_loop:
	FMOVD (R1), F0
	FMOVD (R2), F1
	FMADDD F20, F0, F1, F0
	FMOVD F0, (R0)
	FMADDD F0, F10, F0, F10
	ADD $8, R0, R0
	ADD $8, R1, R1
	ADD $8, R2, R2
	SUBS $1, R3, R3
	BNE hebbn_neon_loop
hebbn_neon_done:
	FSQRTD F10, F10
	FMOVD F10, ret+80(FP)
	RET

// hebbScaleNEON(out []float64, scale float64) — in-place scale
TEXT ·hebbScaleNEON(SB), NOSPLIT, $0-32
	MOVD out+0(FP), R0
	MOVD out_len+8(FP), R3
	FMOVD scale+24(FP), F20
	CBZ R3, hsc_neon_done
hsc_neon_loop:
	FMOVD (R0), F0
	FMULD F20, F0, F0
	FMOVD F0, (R0)
	ADD $8, R0, R0
	SUBS $1, R3, R3
	BNE hsc_neon_loop
hsc_neon_done:
	RET

// ojaStepNEON(out, params, grads []float64, lr, postSq float64)
TEXT ·ojaStepNEON(SB), NOSPLIT, $0-88
	MOVD out+0(FP), R0
	MOVD params+24(FP), R1
	MOVD grads+48(FP), R2
	MOVD out_len+8(FP), R3
	FMOVD lr+72(FP), F20
	FMOVD postSq+80(FP), F21
	FMULD F20, F21, F21                       // F21 = lr*postSq
	CBZ R3, oja_neon_done
oja_neon_loop:
	FMOVD (R1), F0
	FMOVD (R2), F1
	FMOVD F0, F2
	FMADDD F20, F2, F1, F2                     // params + lr*grads
	FMSUBD F21, F2, F0, F2                     // - lr*postSq*params
	FMOVD F2, (R0)
	ADD $8, R0, R0
	ADD $8, R1, R1
	ADD $8, R2, R2
	SUBS $1, R3, R3
	BNE oja_neon_loop
oja_neon_done:
	RET

// reduceSumSqNEON(a []float64) float64
TEXT ·reduceSumSqNEON(SB), NOSPLIT, $0-32
	MOVD a+0(FP), R0
	MOVD a_len+8(FP), R1
	FMOVD $0.0, F0
	CBZ R1, rss_neon_done
rss_neon_loop:
	FMOVD (R0), F1
	FMADDD F1, F0, F1, F0
	ADD $8, R0, R0
	SUBS $1, R1, R1
	BNE rss_neon_loop
rss_neon_done:
	FMOVD F0, ret+24(FP)
	RET
