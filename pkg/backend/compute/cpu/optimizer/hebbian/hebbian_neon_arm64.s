#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// hebbStepNEON(out, params, grads []float64, lr float64)
TEXT ·hebbStepNEON(SB), NOSPLIT, $8-80
	MOVD out+0(FP), R0
	MOVD params+24(FP), R1
	MOVD grads+48(FP), R2
	MOVD out_len+8(FP), R3
	FMOVD lr+72(FP), F20
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	CBZ R3, hebb_neon_done
	LSR $1, R3, R4
	CBZ R4, hebb_neon_tail
hebb_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R2), [V1.D2]
	VFMUL_D2(20, 1, 1)
	VFADD_D2(1, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R4, R4
	BNE hebb_neon_loop
hebb_neon_tail:
	TST $1, R3
	BEQ hebb_neon_done
	FMOVD (R1), F0
	FMOVD (R2), F1
	FMADDD F20, F0, F1, F0
	FMOVD F0, (R0)
hebb_neon_done:
	RET

// hebbStepNormNEON(out, params, grads []float64, lr float64) float64
// returns L2 norm of updated weights.
TEXT ·hebbStepNormNEON(SB), NOSPLIT, $24-88
	MOVD out+0(FP), R0
	MOVD params+24(FP), R1
	MOVD grads+48(FP), R2
	MOVD out_len+8(FP), R3
	FMOVD lr+72(FP), F20
	FMOVD F20, 16(RSP)
	ADD $16, RSP, R5
	VLD1R (R5), [V20.D2]
	FMOVD $0.0, F10
	VEOR V10.B16, V10.B16, V10.B16
	CBZ R3, hebbn_neon_done
	LSR $1, R3, R4
	CBZ R4, hebbn_neon_tail
hebbn_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R2), [V1.D2]
	VFMUL_D2(20, 1, 1)
	VFADD_D2(1, 0, 0)
	VST1.P [V0.D2], 16(R0)
	VFMUL_D2(0, 0, 2)
	VFADD_D2(2, 10, 10)
	SUBS $1, R4, R4
	BNE hebbn_neon_loop
	MOVD RSP, R4
	VST1.P [V10.D2], 16(R4)
	FMOVD 0(RSP), F10
	FMOVD 8(RSP), F11
	FADDD F11, F10, F10
hebbn_neon_tail:
	TST $1, R3
	BEQ hebbn_neon_done
	FMOVD (R1), F0
	FMOVD (R2), F1
	FMADDD F20, F0, F1, F0
	FMOVD F0, (R0)
	FMADDD F0, F10, F0, F10
hebbn_neon_done:
	FSQRTD F10, F10
	FMOVD F10, ret+80(FP)
	RET

// hebbScaleNEON(out []float64, scale float64) — in-place scale
TEXT ·hebbScaleNEON(SB), NOSPLIT, $8-32
	MOVD out+0(FP), R0
	MOVD out_len+8(FP), R3
	FMOVD scale+24(FP), F20
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	CBZ R3, hsc_neon_done
	LSR $1, R3, R4
	CBZ R4, hsc_neon_tail
hsc_neon_loop:
	VLD1.P 16(R0), [V0.D2]
	VFMUL_D2(20, 0, 0)
	SUB $16, R0, R0
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R4, R4
	BNE hsc_neon_loop
hsc_neon_tail:
	TST $1, R3
	BEQ hsc_neon_done
	FMOVD (R0), F0
	FMULD F20, F0, F0
	FMOVD F0, (R0)
hsc_neon_done:
	RET

// ojaStepNEON(out, params, grads []float64, lr, postSq float64)
TEXT ·ojaStepNEON(SB), NOSPLIT, $16-88
	MOVD out+0(FP), R0
	MOVD params+24(FP), R1
	MOVD grads+48(FP), R2
	MOVD out_len+8(FP), R3
	FMOVD lr+72(FP), F20
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	FMOVD postSq+80(FP), F21
	FMULD F20, F21, F21                       // F21 = lr*postSq
	FMOVD F21, 8(RSP)
	ADD $8, RSP, R5
	VLD1R (R5), [V21.D2]
	CBZ R3, oja_neon_done
	LSR $1, R3, R4
	CBZ R4, oja_neon_tail
oja_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R2), [V1.D2]
	VFMUL_D2(20, 1, 2)
	VFADD_D2(0, 2, 2)
	VFMUL_D2(21, 0, 3)
	VFSUB_D2(3, 2, 2)
	VST1.P [V2.D2], 16(R0)
	SUBS $1, R4, R4
	BNE oja_neon_loop
oja_neon_tail:
	TST $1, R3
	BEQ oja_neon_done
	FMOVD (R1), F0
	FMOVD (R2), F1
	FMOVD F0, F2
	FMADDD F20, F2, F1, F2                     // params + lr*grads
	FMSUBD F21, F2, F0, F2                     // - lr*postSq*params
	FMOVD F2, (R0)
oja_neon_done:
	RET

// reduceSumSqNEON(a []float64) float64
TEXT ·reduceSumSqNEON(SB), NOSPLIT, $16-32
	MOVD a+0(FP), R0
	MOVD a_len+8(FP), R1
	FMOVD $0.0, F0
	VEOR V0.B16, V0.B16, V0.B16
	CBZ R1, rss_neon_done
	LSR $1, R1, R2
	CBZ R2, rss_neon_tail
rss_neon_loop:
	VLD1.P 16(R0), [V1.D2]
	VFMUL_D2(1, 1, 1)
	VFADD_D2(1, 0, 0)
	SUBS $1, R2, R2
	BNE rss_neon_loop
	MOVD RSP, R2
	VST1.P [V0.D2], 16(R2)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0
rss_neon_tail:
	TST $1, R1
	BEQ rss_neon_done
	FMOVD (R0), F1
	FMADDD F1, F0, F1, F0
rss_neon_done:
	FMOVD F0, ret+24(FP)
	RET
