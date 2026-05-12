#include "textflag.h"

// larsStepNEON(out, velocity, params, grads []float64,
//              localLR, momentum, wd float64)
TEXT ·larsStepNEON(SB), NOSPLIT, $0-120
	MOVD out+0(FP), R0
	MOVD velocity+24(FP), R1
	MOVD params+48(FP), R2
	MOVD grads+72(FP), R3
	MOVD out_len+8(FP), R4
	FMOVD localLR+96(FP), F20
	FMOVD momentum+104(FP), F21
	FMOVD wd+112(FP), F22

	CBZ R4, lars_neon_done
lars_neon_loop:
	FMOVD (R1), F0
	FMOVD (R2), F1
	FMOVD (R3), F2

	// effGrad = g + wd*params
	FMADDD F22, F2, F1, F3                     // F3 = wd*p + g
	// v = μ*v + localLR*effGrad
	FMULD F21, F0, F0
	FMADDD F20, F0, F3, F0
	FMOVD F0, (R1)
	// out = params - v
	FSUBD F0, F1, F1
	FMOVD F1, (R0)

	ADD $8, R0, R0
	ADD $8, R1, R1
	ADD $8, R2, R2
	ADD $8, R3, R3
	SUBS $1, R4, R4
	BNE lars_neon_loop

lars_neon_done:
	RET

// lambEMANEON(m, v, grads []float64, beta1, oneMinusBeta1, beta2, oneMinusBeta2 float64)
TEXT ·lambEMANEON(SB), NOSPLIT, $0-104
	MOVD m+0(FP), R0
	MOVD v+24(FP), R1
	MOVD grads+48(FP), R2
	MOVD m_len+8(FP), R3
	FMOVD beta1+72(FP), F20
	FMOVD oneMinusBeta1+80(FP), F21
	FMOVD beta2+88(FP), F22
	FMOVD oneMinusBeta2+96(FP), F23

	CBZ R3, lema_neon_done
lema_neon_loop:
	FMOVD (R0), F0
	FMOVD (R1), F1
	FMOVD (R2), F2

	FMULD F2, F2, F3
	FMULD F20, F0, F0
	FMADDD F21, F0, F2, F0
	FMOVD F0, (R0)
	FMULD F22, F1, F1
	FMADDD F23, F1, F3, F1
	FMOVD F1, (R1)

	ADD $8, R0, R0
	ADD $8, R1, R1
	ADD $8, R2, R2
	SUBS $1, R3, R3
	BNE lema_neon_loop

lema_neon_done:
	RET

// lambL2NormSqNEON(a []float64) float64
TEXT ·lambL2NormSqNEON(SB), NOSPLIT, $0-32
	MOVD a+0(FP), R0
	MOVD a_len+8(FP), R1
	FMOVD $0.0, F0
	CBZ R1, ll2_neon_done
ll2_neon_loop:
	FMOVD (R0), F1
	FMADDD F1, F0, F1, F0
	ADD $8, R0, R0
	SUBS $1, R1, R1
	BNE ll2_neon_loop
ll2_neon_done:
	FMOVD F0, ret+24(FP)
	RET

// lambUpdateNormSqNEON(m, v, params []float64, bc1Inv, bc2Inv, eps, wd float64) float64
TEXT ·lambUpdateNormSqNEON(SB), NOSPLIT, $0-112
	MOVD m+0(FP), R0
	MOVD v+24(FP), R1
	MOVD params+48(FP), R2
	MOVD m_len+8(FP), R3
	FMOVD bc1Inv+72(FP), F20
	FMOVD bc2Inv+80(FP), F21
	FMOVD eps+88(FP), F22
	FMOVD wd+96(FP), F23
	FMOVD $0.0, F10

	CBZ R3, luns_neon_done
luns_neon_loop:
	FMOVD (R0), F0
	FMOVD (R1), F1
	FMOVD (R2), F2

	FMULD F20, F0, F3                          // mHat
	FMULD F21, F1, F4                          // vHat
	FSQRTD F4, F5
	FADDD F22, F5, F5
	FDIVD F5, F3, F6                           // mHat/denom
	FMADDD F23, F6, F2, F6                     // + wd*params  (FMADDD A,B,C,D: D=A*C+B → F6 = wd*params + F6? Need wd*p+F6. Yes: A=F23(wd), B=F6(curr), C=F2(p))
	FMADDD F6, F10, F6, F10                    // acc += val²

	ADD $8, R0, R0
	ADD $8, R1, R1
	ADD $8, R2, R2
	SUBS $1, R3, R3
	BNE luns_neon_loop

luns_neon_done:
	FMOVD F10, ret+104(FP)
	RET

// lambStepNEON(out, m, v, params, grads []float64,
//              ratio, bc1Inv, bc2Inv, eps, wd float64)
TEXT ·lambStepNEON(SB), NOSPLIT, $0-160
	MOVD out+0(FP), R0
	MOVD m+24(FP), R1
	MOVD v+48(FP), R7
	MOVD params+72(FP), R2
	MOVD grads+96(FP), R3
	MOVD out_len+8(FP), R4

	FMOVD ratio+120(FP), F20
	FMOVD bc1Inv+128(FP), F21
	FMOVD bc2Inv+136(FP), F22
	FMOVD eps+144(FP), F23
	FMOVD wd+152(FP), F24

	CBZ R4, lamb_neon_done
lamb_neon_loop:
	FMOVD (R1), F0
	FMOVD (R7), F1
	FMOVD (R2), F2

	FMULD F21, F0, F3
	FMULD F22, F1, F4
	FSQRTD F4, F5
	FADDD F23, F5, F5
	FDIVD F5, F3, F6
	FMADDD F24, F6, F2, F6                     // update = mHat/denom + wd*params
	FMSUBD F20, F2, F6, F2                     // out = params - ratio*update
	FMOVD F2, (R0)

	ADD $8, R0, R0
	ADD $8, R1, R1
	ADD $8, R7, R7
	ADD $8, R2, R2
	ADD $8, R3, R3
	SUBS $1, R4, R4
	BNE lamb_neon_loop

lamb_neon_done:
	RET
