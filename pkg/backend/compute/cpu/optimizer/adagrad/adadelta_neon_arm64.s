#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFDIV_D2(m, n, d) WORD $(0x6E60FC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFSQRT_D2(n, d) WORD $(0x6EE1F800 | ((n) << 5) | (d))

// adadeltaStepNEON(out, eg2, edp2, params, grads []float64,
//                  rho, oneMinusRho, eps, wd float64)
TEXT ·adadeltaStepNEON(SB), NOSPLIT, $32-152
	MOVD out+0(FP), R0
	MOVD eg2+24(FP), R1
	MOVD edp2+48(FP), R2
	MOVD params+72(FP), R3
	MOVD grads+96(FP), R4
	MOVD out_len+8(FP), R5

	FMOVD rho+120(FP), F20
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	FMOVD oneMinusRho+128(FP), F21
	FMOVD F21, 8(RSP)
	ADD $8, RSP, R7
	VLD1R (R7), [V21.D2]
	FMOVD eps+136(FP), F22
	FMOVD F22, 16(RSP)
	ADD $16, RSP, R7
	VLD1R (R7), [V22.D2]
	FMOVD wd+144(FP), F23
	FMOVD F23, 24(RSP)
	ADD $24, RSP, R7
	VLD1R (R7), [V23.D2]
	VEOR V31.B16, V31.B16, V31.B16

	LSR  $1, R5, R6
	CBZ  R6, ad_neon_tail
ad_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R2), [V1.D2]
	VLD1.P 16(R3), [V2.D2]
	VLD1.P 16(R4), [V3.D2]
	VFMUL_D2(23, 2, 4)
	VFADD_D2(4, 3, 4)
	VFMUL_D2(20, 0, 0)
	VFMUL_D2(4, 4, 5)
	VFMUL_D2(21, 5, 5)
	VFADD_D2(5, 0, 0)
	SUB $16, R1, R1
	VST1.P [V0.D2], 16(R1)

	VFADD_D2(22, 1, 6)
	VFSQRT_D2(6, 6)
	VFADD_D2(22, 0, 7)
	VFSQRT_D2(7, 7)
	VFDIV_D2(7, 6, 8)
	VFMUL_D2(4, 8, 8)
	VFSUB_D2(8, 31, 8)

	VFMUL_D2(20, 1, 1)
	VFMUL_D2(8, 8, 9)
	VFMUL_D2(21, 9, 9)
	VFADD_D2(9, 1, 1)
	SUB $16, R2, R2
	VST1.P [V1.D2], 16(R2)
	VFADD_D2(8, 2, 2)
	VST1.P [V2.D2], 16(R0)
	SUBS $1, R6, R6
	BNE  ad_neon_loop

ad_neon_tail:
	AND $1, R5, R7
	CBZ R7, ad_neon_done
	FMOVD (R1), F0
	FMOVD (R2), F2
	FMOVD (R3), F4
	FMOVD (R4), F6
	FMADDD F23, F6, F4, F6
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
