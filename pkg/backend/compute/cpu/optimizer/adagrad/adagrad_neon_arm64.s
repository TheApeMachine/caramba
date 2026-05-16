#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFDIV_D2(m, n, d) WORD $(0x6E60FC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFSQRT_D2(n, d) WORD $(0x6EE1F800 | ((n) << 5) | (d))

// adagradStepNEON(out, G, params, grads []float64, lr, eps, wd float64)
TEXT ·adagradStepNEON(SB), NOSPLIT, $24-120
	MOVD out+0(FP), R0
	MOVD G+24(FP), R1
	MOVD params+48(FP), R2
	MOVD grads+72(FP), R3
	MOVD out_len+8(FP), R4

	FMOVD lr+96(FP), F20
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	FMOVD eps+104(FP), F21
	FMOVD F21, 8(RSP)
	ADD $8, RSP, R6
	VLD1R (R6), [V21.D2]
	FMOVD wd+112(FP), F22
	FMOVD F22, 16(RSP)
	ADD $16, RSP, R6
	VLD1R (R6), [V22.D2]

	LSR  $1, R4, R5
	CBZ  R5, ag_neon_tail
ag_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R2), [V1.D2]
	VLD1.P 16(R3), [V2.D2]
	VFMUL_D2(22, 1, 3)
	VFADD_D2(3, 2, 3)
	VFMUL_D2(3, 3, 4)
	VFADD_D2(4, 0, 0)
	SUB $16, R1, R1
	VST1.P [V0.D2], 16(R1)
	VFSQRT_D2(0, 4)
	VFADD_D2(21, 4, 4)
	VFDIV_D2(4, 3, 5)
	VFMUL_D2(20, 5, 5)
	VFSUB_D2(5, 1, 1)
	VST1.P [V1.D2], 16(R0)
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
