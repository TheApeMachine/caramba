#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// sgdVanillaNEON(out, params, grads []float64, lr, wd float64)
TEXT ·sgdVanillaNEON(SB), NOSPLIT, $16-88
	MOVD out+0(FP), R0
	MOVD params+24(FP), R1
	MOVD grads+48(FP), R2
	MOVD out_len+8(FP), R3
	FMOVD lr+72(FP), F20
	FMOVD wd+80(FP), F21
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	FMOVD F21, 8(RSP)
	ADD $8, RSP, R6
	VLD1R (R6), [V21.D2]

	LSR  $1, R3, R4
	CBZ  R4, sgdv_neon_tail
sgdv_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R2), [V2.D2]
	VFMUL_D2(21, 0, 3)
	VFADD_D2(2, 3, 3)
	VFMUL_D2(20, 3, 3)
	VFSUB_D2(3, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R4, R4
	BNE  sgdv_neon_loop

sgdv_neon_tail:
	AND $1, R3, R5
	CBZ R5, sgdv_neon_done
	FMOVD (R1), F0
	FMOVD (R2), F2
	FMADDD F21, F2, F0, F2
	FMSUBD F20, F0, F2, F0
	FMOVD F0, (R0)
sgdv_neon_done:
	RET

// sgdMomentumNEON(out, params, grads, velocity []float64,
//                 lr, wd, momentum float64, nesterov uint64)
TEXT ·sgdMomentumNEON(SB), NOSPLIT, $24-128
	MOVD out+0(FP), R0
	MOVD params+24(FP), R1
	MOVD grads+48(FP), R2
	MOVD velocity+72(FP), R3
	MOVD out_len+8(FP), R4
	FMOVD lr+96(FP), F20
	FMOVD wd+104(FP), F21
	FMOVD momentum+112(FP), F22
	MOVD nesterov+120(FP), R12
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	FMOVD F21, 8(RSP)
	ADD $8, RSP, R13
	VLD1R (R13), [V21.D2]
	FMOVD F22, 16(RSP)
	ADD $16, RSP, R13
	VLD1R (R13), [V22.D2]

	LSR  $1, R4, R5
	CBZ  R5, sgdm_neon_tail
sgdm_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R2), [V2.D2]
	VLD1.P 16(R3), [V4.D2]

	VFMUL_D2(21, 0, 6)
	VFADD_D2(2, 6, 6)
	VFMUL_D2(22, 4, 4)
	VFADD_D2(6, 4, 4)
	SUB $16, R3, R3
	VST1.P [V4.D2], 16(R3)

	CBZ R12, sgdm_neon_addV
	VFMUL_D2(22, 4, 8)
	VFADD_D2(6, 8, 8)
	VFMUL_D2(20, 8, 8)
	VFSUB_D2(8, 0, 6)
	B sgdm_neon_store
sgdm_neon_addV:
	VFMUL_D2(20, 4, 8)
	VFSUB_D2(8, 0, 6)
sgdm_neon_store:
	VST1.P [V6.D2], 16(R0)
	SUBS $1, R5, R5
	BNE  sgdm_neon_loop

sgdm_neon_tail:
	AND $1, R4, R6
	CBZ R6, sgdm_neon_done
	FMOVD (R1), F0
	FMOVD (R2), F2
	FMOVD (R3), F4

	FMULD F21, F0, F6
	FADDD F2, F6, F6
	FMULD F22, F4, F4
	FADDD F6, F4, F4
	FMOVD F4, (R3)

	CBZ R12, sgdm_neon_addV2
	FMULD F22, F4, F8
	FADDD F6, F8, F8
	FMULD F20, F8, F8
	FSUBD F8, F0, F6
	B sgdm_neon_storeTail
sgdm_neon_addV2:
	FMULD F20, F4, F8
	FSUBD F8, F0, F6
sgdm_neon_storeTail:
	FMOVD F6, (R0)

sgdm_neon_done:
	RET
