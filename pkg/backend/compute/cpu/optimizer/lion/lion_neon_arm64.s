#include "textflag.h"

DATA ·lionOne+0(SB)/8, $1.0
GLOBL ·lionOne(SB), RODATA, $8
DATA ·lionNegOne+0(SB)/8, $-1.0
GLOBL ·lionNegOne(SB), RODATA, $8

// lionStepNEON(out, m, params, grads []float64,
//              lr, beta1, oneMinusBeta1, beta2, oneMinusBeta2, wd float64)
TEXT ·lionStepNEON(SB), NOSPLIT, $0-144
	MOVD out+0(FP), R0
	MOVD m+24(FP), R1
	MOVD params+48(FP), R2
	MOVD grads+72(FP), R3
	MOVD out_len+8(FP), R4

	FMOVD lr+96(FP), F20
	FMOVD beta1+104(FP), F21
	FMOVD oneMinusBeta1+112(FP), F22
	FMOVD beta2+120(FP), F23
	FMOVD oneMinusBeta2+128(FP), F24
	FMOVD wd+136(FP), F25
	FMOVD ·lionOne(SB), F26
	FMOVD ·lionNegOne(SB), F27
	FMOVD $0.0, F28                            // zero

	CBZ R4, lion_neon_done
lion_neon_loop:
	FMOVD (R1), F0                             // m
	FMOVD (R2), F1                             // params
	FMOVD (R3), F2                             // grads

	// interp = β1*m + (1-β1)*g
	FMULD F21, F0, F3
	FMADDD F22, F3, F2, F3

	// sign(interp)
	FCMPD F28, F3
	BEQ lion_neon_zero
	BMI lion_neon_neg
	// positive
	FMOVD F26, F4
	B lion_neon_have_sign
lion_neon_neg:
	FMOVD F27, F4
	B lion_neon_have_sign
lion_neon_zero:
	FMOVD F28, F4
lion_neon_have_sign:
	// out = params - lr*sign - lr*wd*params
	FMSUBD F20, F1, F4, F5                     // F5 = params - lr*sign
	FMULD F25, F1, F6
	FMSUBD F20, F5, F6, F5                     // F5 -= lr*wd*params
	FMOVD F5, (R0)

	// m = β2*m + (1-β2)*g
	FMULD F23, F0, F0
	FMADDD F24, F0, F2, F0
	FMOVD F0, (R1)

	ADD $8, R0, R0
	ADD $8, R1, R1
	ADD $8, R2, R2
	ADD $8, R3, R3
	SUBS $1, R4, R4
	BNE  lion_neon_loop

lion_neon_done:
	RET
