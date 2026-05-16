#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFCMGT_D2(m, n, d) WORD $(0x6EE0E400 | ((m) << 16) | ((n) << 5) | (d))

DATA ·lionOne+0(SB)/8, $1.0
GLOBL ·lionOne(SB), RODATA, $8
DATA ·lionNegOne+0(SB)/8, $-1.0
GLOBL ·lionNegOne(SB), RODATA, $8

// lionStepNEON(out, m, params, grads []float64,
//              lr, beta1, oneMinusBeta1, beta2, oneMinusBeta2, wd float64)
TEXT ·lionStepNEON(SB), NOSPLIT, $8-144
	MOVD out+0(FP), R0
	MOVD m+24(FP), R1
	MOVD params+48(FP), R2
	MOVD grads+72(FP), R3
	MOVD out_len+8(FP), R4

	FMOVD lr+96(FP), F20
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	FMOVD beta1+104(FP), F21
	FMOVD F21, 0(RSP)
	VLD1R (RSP), [V21.D2]
	FMOVD oneMinusBeta1+112(FP), F22
	FMOVD F22, 0(RSP)
	VLD1R (RSP), [V22.D2]
	FMOVD beta2+120(FP), F23
	FMOVD F23, 0(RSP)
	VLD1R (RSP), [V23.D2]
	FMOVD oneMinusBeta2+128(FP), F24
	FMOVD F24, 0(RSP)
	VLD1R (RSP), [V24.D2]
	FMOVD wd+136(FP), F25
	FMOVD F25, 0(RSP)
	VLD1R (RSP), [V25.D2]
	FMOVD ·lionOne(SB), F26
	FMOVD F26, 0(RSP)
	VLD1R (RSP), [V26.D2]
	FMOVD ·lionNegOne(SB), F27
	FMOVD F27, 0(RSP)
	VLD1R (RSP), [V27.D2]
	FMOVD $0.0, F28                            // zero
	VEOR V28.B16, V28.B16, V28.B16

	CBZ R4, lion_neon_done
	CMP $2, R4
	BLT lion_neon_tail
lion_neon_loop:
	VLD1.P 16(R1), [V0.D2]                     // m
	VLD1.P 16(R2), [V1.D2]                     // params
	VLD1.P 16(R3), [V2.D2]                     // grads

	VFMUL_D2(21, 0, 3)
	VFMUL_D2(22, 2, 4)
	VFADD_D2(4, 3, 3)                          // interp

	VFCMGT_D2(28, 3, 4)                        // interp > 0
	VFCMGT_D2(3, 28, 5)                        // interp < 0
	VAND V26.B16, V4.B16, V4.B16
	VAND V27.B16, V5.B16, V5.B16
	VORR  V5.B16, V4.B16, V4.B16               // sign(interp)

	VFMUL_D2(20, 4, 5)
	VFSUB_D2(5, 1, 6)
	VFMUL_D2(25, 1, 7)
	VFMUL_D2(20, 7, 7)
	VFSUB_D2(7, 6, 6)                          // out
	VST1.P [V6.D2], 16(R0)

	VFMUL_D2(23, 0, 0)
	VFMUL_D2(24, 2, 7)
	VFADD_D2(7, 0, 0)
	SUB $16, R1, R1
	VST1.P [V0.D2], 16(R1)

	SUBS $2, R4, R4
	CMP  $2, R4
	BGE  lion_neon_loop

lion_neon_tail:
	CBZ R4, lion_neon_done
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

lion_neon_done:
	RET
