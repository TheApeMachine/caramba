#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFDIV_D2(m, n, d) WORD $(0x6E60FC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFSQRT_D2(n, d) WORD $(0x6EE1F800 | ((n) << 5) | (d))

// rmspropCenteredMomentumNEON(out, v, gAvg, buf, params, grads []float64,
//                             lr, alpha, oneMinusAlpha, eps, momentum, wd float64)
TEXT ·rmspropCenteredMomentumNEON(SB), NOSPLIT, $48-192
	MOVD out+0(FP), R0
	MOVD v+24(FP), R1
	MOVD gAvg+48(FP), R7
	MOVD buf+72(FP), R8
	MOVD params+96(FP), R2
	MOVD grads+120(FP), R3
	MOVD out_len+8(FP), R4

	FMOVD lr+144(FP), F20
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	FMOVD alpha+152(FP), F21
	FMOVD F21, 8(RSP)
	ADD $8, RSP, R6
	VLD1R (R6), [V21.D2]
	FMOVD oneMinusAlpha+160(FP), F22
	FMOVD F22, 16(RSP)
	ADD $16, RSP, R6
	VLD1R (R6), [V22.D2]
	FMOVD eps+168(FP), F23
	FMOVD F23, 24(RSP)
	ADD $24, RSP, R6
	VLD1R (R6), [V23.D2]
	FMOVD momentum+176(FP), F24
	FMOVD F24, 32(RSP)
	ADD $32, RSP, R6
	VLD1R (R6), [V24.D2]
	FMOVD wd+184(FP), F25
	FMOVD F25, 40(RSP)
	ADD $40, RSP, R6
	VLD1R (R6), [V25.D2]

	LSR $1, R4, R5
	CBZ R5, rmscm_neon_tail
rmscm_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R7), [V1.D2]
	VLD1.P 16(R8), [V14.D2]
	VLD1.P 16(R2), [V2.D2]
	VLD1.P 16(R3), [V3.D2]

	VFMUL_D2(25, 2, 4)
	VFADD_D2(4, 3, 4)                         // geff
	VFMUL_D2(21, 0, 0)
	VFMUL_D2(4, 4, 5)
	VFMUL_D2(22, 5, 5)
	VFADD_D2(5, 0, 0)
	SUB $16, R1, R1
	VST1.P [V0.D2], 16(R1)

	VFMUL_D2(21, 1, 1)
	VFMUL_D2(22, 4, 5)
	VFADD_D2(5, 1, 1)
	SUB $16, R7, R7
	VST1.P [V1.D2], 16(R7)

	VFMUL_D2(1, 1, 6)
	VFSUB_D2(6, 0, 6)
	VFSQRT_D2(6, 6)
	VFADD_D2(23, 6, 6)
	VFDIV_D2(6, 4, 7)

	VFMUL_D2(24, 14, 14)
	VFADD_D2(7, 14, 14)                       // buf = momentum*buf + upd
	SUB $16, R8, R8
	VST1.P [V14.D2], 16(R8)

	VFMUL_D2(20, 14, 7)
	VFSUB_D2(7, 2, 2)
	VST1.P [V2.D2], 16(R0)

	SUBS $1, R5, R5
	BNE rmscm_neon_loop

rmscm_neon_tail:
	AND $1, R4, R6
	CBZ R6, rmscm_neon_done
	FMOVD (R1), F0
	FMOVD (R7), F1
	FMOVD (R8), F14
	FMOVD (R2), F2
	FMOVD (R3), F3

	FMADDD F25, F3, F2, F4
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
	FADDD F7, F14, F14
	FMOVD F14, (R8)

	FMSUBD F20, F2, F14, F2
	FMOVD F2, (R0)

rmscm_neon_done:
	RET
