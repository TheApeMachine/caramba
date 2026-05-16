#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFDIV_D2(m, n, d) WORD $(0x6E60FC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFSQRT_D2(n, d) WORD $(0x6EE1F800 | ((n) << 5) | (d))

// rmspropPlainNEON(out, v, params, grads []float64,
//                  lr, alpha, oneMinusAlpha, eps, wd float64)
TEXT ·rmspropPlainNEON(SB), NOSPLIT, $40-136
	MOVD out+0(FP), R0
	MOVD v+24(FP), R1
	MOVD params+48(FP), R2
	MOVD grads+72(FP), R3
	MOVD out_len+8(FP), R4

	FMOVD lr+96(FP), F20
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	FMOVD alpha+104(FP), F21
	FMOVD F21, 8(RSP)
	ADD $8, RSP, R6
	VLD1R (R6), [V21.D2]
	FMOVD oneMinusAlpha+112(FP), F22
	FMOVD F22, 16(RSP)
	ADD $16, RSP, R6
	VLD1R (R6), [V22.D2]
	FMOVD eps+120(FP), F23
	FMOVD F23, 24(RSP)
	ADD $24, RSP, R6
	VLD1R (R6), [V23.D2]
	FMOVD wd+128(FP), F24
	FMOVD F24, 32(RSP)
	ADD $32, RSP, R6
	VLD1R (R6), [V24.D2]

	LSR $1, R4, R5
	CBZ R5, rmsp_neon_tail
rmsp_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R2), [V2.D2]
	VLD1.P 16(R3), [V4.D2]

	VFMUL_D2(24, 2, 6)
	VFADD_D2(6, 4, 6)                         // geff
	VFMUL_D2(21, 0, 0)
	VFMUL_D2(6, 6, 8)
	VFMUL_D2(22, 8, 8)
	VFADD_D2(8, 0, 0)                         // v
	SUB $16, R1, R1
	VST1.P [V0.D2], 16(R1)

	VFSQRT_D2(0, 10)
	VFADD_D2(23, 10, 10)
	VFDIV_D2(10, 6, 12)
	VFMUL_D2(20, 12, 12)
	VFSUB_D2(12, 2, 2)
	VST1.P [V2.D2], 16(R0)

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
