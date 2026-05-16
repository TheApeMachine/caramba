#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFDIV_D2(m, n, d) WORD $(0x6E60FC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFSQRT_D2(n, d) WORD $(0x6EE1F800 | ((n) << 5) | (d))

// larsStepNEON(out, velocity, params, grads []float64,
//              localLR, momentum, wd float64)
TEXT ·larsStepNEON(SB), NOSPLIT, $24-120
	MOVD out+0(FP), R0
	MOVD velocity+24(FP), R1
	MOVD params+48(FP), R2
	MOVD grads+72(FP), R3
	MOVD out_len+8(FP), R4
	FMOVD localLR+96(FP), F20
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	FMOVD momentum+104(FP), F21
	FMOVD F21, 8(RSP)
	ADD $8, RSP, R6
	VLD1R (R6), [V21.D2]
	FMOVD wd+112(FP), F22
	FMOVD F22, 16(RSP)
	ADD $16, RSP, R6
	VLD1R (R6), [V22.D2]

	LSR $1, R4, R5
	CBZ R5, lars_neon_tail
lars_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R2), [V1.D2]
	VLD1.P 16(R3), [V2.D2]
	VFMUL_D2(22, 1, 3)
	VFADD_D2(3, 2, 3)
	VFMUL_D2(21, 0, 0)
	VFMUL_D2(20, 3, 3)
	VFADD_D2(3, 0, 0)
	SUB $16, R1, R1
	VST1.P [V0.D2], 16(R1)
	VFSUB_D2(0, 1, 1)
	VST1.P [V1.D2], 16(R0)
	SUBS $1, R5, R5
	BNE lars_neon_loop
lars_neon_tail:
	AND $1, R4, R6
	CBZ R6, lars_neon_done
	FMOVD (R1), F0
	FMOVD (R2), F1
	FMOVD (R3), F2
	FMADDD F22, F2, F1, F3
	FMULD F21, F0, F0
	FMADDD F20, F0, F3, F0
	FMOVD F0, (R1)
	FSUBD F0, F1, F1
	FMOVD F1, (R0)
lars_neon_done:
	RET

// lambEMANEON(m, v, grads []float64, beta1, oneMinusBeta1, beta2, oneMinusBeta2 float64)
TEXT ·lambEMANEON(SB), NOSPLIT, $32-104
	MOVD m+0(FP), R0
	MOVD v+24(FP), R1
	MOVD grads+48(FP), R2
	MOVD m_len+8(FP), R3
	FMOVD beta1+72(FP), F20
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	FMOVD oneMinusBeta1+80(FP), F21
	FMOVD F21, 8(RSP)
	ADD $8, RSP, R4
	VLD1R (R4), [V21.D2]
	FMOVD beta2+88(FP), F22
	FMOVD F22, 16(RSP)
	ADD $16, RSP, R4
	VLD1R (R4), [V22.D2]
	FMOVD oneMinusBeta2+96(FP), F23
	FMOVD F23, 24(RSP)
	ADD $24, RSP, R4
	VLD1R (R4), [V23.D2]

	LSR $1, R3, R4
	CBZ R4, lema_neon_tail
lema_neon_loop:
	VLD1.P 16(R0), [V0.D2]
	VLD1.P 16(R1), [V1.D2]
	VLD1.P 16(R2), [V2.D2]
	VFMUL_D2(20, 0, 0)
	VFMUL_D2(21, 2, 3)
	VFADD_D2(3, 0, 0)
	SUB $16, R0, R0
	VST1.P [V0.D2], 16(R0)
	VFMUL_D2(22, 1, 1)
	VFMUL_D2(2, 2, 3)
	VFMUL_D2(23, 3, 3)
	VFADD_D2(3, 1, 1)
	SUB $16, R1, R1
	VST1.P [V1.D2], 16(R1)
	SUBS $1, R4, R4
	BNE lema_neon_loop
lema_neon_tail:
	AND $1, R3, R4
	CBZ R4, lema_neon_done
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
lema_neon_done:
	RET

// lambL2NormSqNEON(a []float64) float64
TEXT ·lambL2NormSqNEON(SB), NOSPLIT, $16-32
	MOVD a+0(FP), R0
	MOVD a_len+8(FP), R1
	VEOR V0.B16, V0.B16, V0.B16
	LSR  $1, R1, R2
	CBZ  R2, ll2_neon_tail
ll2_neon_loop:
	VLD1.P 16(R0), [V1.D2]
	VFMUL_D2(1, 1, 1)
	VFADD_D2(1, 0, 0)
	SUBS $1, R2, R2
	BNE ll2_neon_loop
	MOVD RSP, R2
	VST1.P [V0.D2], 16(R2)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0
ll2_neon_tail:
	TST $1, R1
	BEQ ll2_neon_done
	FMOVD (R0), F1
	FMADDD F1, F0, F1, F0
ll2_neon_done:
	FMOVD F0, ret+24(FP)
	RET

// lambUpdateNormSqNEON(m, v, params []float64, bc1Inv, bc2Inv, eps, wd float64) float64
TEXT ·lambUpdateNormSqNEON(SB), NOSPLIT, $48-112
	MOVD m+0(FP), R0
	MOVD v+24(FP), R1
	MOVD params+48(FP), R2
	MOVD m_len+8(FP), R3
	FMOVD bc1Inv+72(FP), F20
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	FMOVD bc2Inv+80(FP), F21
	FMOVD F21, 8(RSP)
	ADD $8, RSP, R4
	VLD1R (R4), [V21.D2]
	FMOVD eps+88(FP), F22
	FMOVD F22, 16(RSP)
	ADD $16, RSP, R4
	VLD1R (R4), [V22.D2]
	FMOVD wd+96(FP), F23
	FMOVD F23, 24(RSP)
	ADD $24, RSP, R4
	VLD1R (R4), [V23.D2]
	VEOR V10.B16, V10.B16, V10.B16

	LSR $1, R3, R4
	CBZ R4, luns_neon_tail
luns_neon_loop:
	VLD1.P 16(R0), [V0.D2]
	VLD1.P 16(R1), [V1.D2]
	VLD1.P 16(R2), [V2.D2]
	VFMUL_D2(20, 0, 3)
	VFMUL_D2(21, 1, 4)
	VFSQRT_D2(4, 5)
	VFADD_D2(22, 5, 5)
	VFDIV_D2(5, 3, 6)
	VFMUL_D2(23, 2, 7)
	VFADD_D2(7, 6, 6)
	VFMUL_D2(6, 6, 7)
	VFADD_D2(7, 10, 10)
	SUBS $1, R4, R4
	BNE luns_neon_loop
	MOVD RSP, R4
	VST1.P [V10.D2], 16(R4)
	FMOVD 0(RSP), F10
	FMOVD 8(RSP), F11
	FADDD F11, F10, F10
luns_neon_tail:
	TST $1, R3
	BEQ luns_neon_done
	FMOVD (R0), F0
	FMOVD (R1), F1
	FMOVD (R2), F2
	FMULD F20, F0, F3
	FMULD F21, F1, F4
	FSQRTD F4, F5
	FADDD F22, F5, F5
	FDIVD F5, F3, F6
	FMADDD F23, F6, F2, F6
	FMADDD F6, F10, F6, F10
luns_neon_done:
	FMOVD F10, ret+104(FP)
	RET

// lambStepNEON(out, m, v, params, grads []float64,
//              ratio, bc1Inv, bc2Inv, eps, wd float64)
TEXT ·lambStepNEON(SB), NOSPLIT, $40-160
	MOVD out+0(FP), R0
	MOVD m+24(FP), R1
	MOVD v+48(FP), R7
	MOVD params+72(FP), R2
	MOVD out_len+8(FP), R4

	FMOVD ratio+120(FP), F20
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	FMOVD bc1Inv+128(FP), F21
	FMOVD F21, 8(RSP)
	ADD $8, RSP, R5
	VLD1R (R5), [V21.D2]
	FMOVD bc2Inv+136(FP), F22
	FMOVD F22, 16(RSP)
	ADD $16, RSP, R5
	VLD1R (R5), [V22.D2]
	FMOVD eps+144(FP), F23
	FMOVD F23, 24(RSP)
	ADD $24, RSP, R5
	VLD1R (R5), [V23.D2]
	FMOVD wd+152(FP), F24
	FMOVD F24, 32(RSP)
	ADD $32, RSP, R5
	VLD1R (R5), [V24.D2]

	LSR $1, R4, R5
	CBZ R5, lamb_neon_tail
lamb_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R7), [V1.D2]
	VLD1.P 16(R2), [V2.D2]
	VFMUL_D2(21, 0, 3)
	VFMUL_D2(22, 1, 4)
	VFSQRT_D2(4, 5)
	VFADD_D2(23, 5, 5)
	VFDIV_D2(5, 3, 6)
	VFMUL_D2(24, 2, 7)
	VFADD_D2(7, 6, 6)
	VFMUL_D2(20, 6, 6)
	VFSUB_D2(6, 2, 2)
	VST1.P [V2.D2], 16(R0)
	SUBS $1, R5, R5
	BNE lamb_neon_loop
lamb_neon_tail:
	AND $1, R4, R5
	CBZ R5, lamb_neon_done
	FMOVD (R1), F0
	FMOVD (R7), F1
	FMOVD (R2), F2
	FMULD F21, F0, F3
	FMULD F22, F1, F4
	FSQRTD F4, F5
	FADDD F23, F5, F5
	FDIVD F5, F3, F6
	FMADDD F24, F6, F2, F6
	FMSUBD F20, F2, F6, F2
	FMOVD F2, (R0)
lamb_neon_done:
	RET
