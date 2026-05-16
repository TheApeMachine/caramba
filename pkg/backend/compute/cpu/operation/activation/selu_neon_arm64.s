#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFCMEQ_D2(m, n, d) WORD $(0x6E20E400 | ((m) << 16) | ((n) << 5) | (d))
#define VFCMGT_D2(m, n, d) WORD $(0x6EE0E400 | ((m) << 16) | ((n) << 5) | (d))

DATA ·seluScale+0(SB)/8, $1.0507009873554805
GLOBL ·seluScale(SB), RODATA, $8
DATA ·seluScaleAlpha+0(SB)/8, $1.7580993408473766
GLOBL ·seluScaleAlpha(SB), RODATA, $8
DATA ·seluOne+0(SB)/8, $1.0
GLOBL ·seluOne(SB), RODATA, $8

// seluBlendNEON(dst, src, expValues []float64)
TEXT ·seluBlendNEON(SB), NOSPLIT, $24-72
	MOVD dst+0(FP), R0
	MOVD src_base+24(FP), R1
	MOVD src_len+32(FP), R2
	MOVD expValues_base+48(FP), R3

	FMOVD ·seluScale(SB), F20
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	FMOVD ·seluScaleAlpha(SB), F21
	FMOVD F21, 8(RSP)
	ADD $8, RSP, R5
	VLD1R (R5), [V21.D2]
	FMOVD ·seluOne(SB), F22
	FMOVD F22, 16(RSP)
	ADD $16, RSP, R5
	VLD1R (R5), [V22.D2]
	VEOR V31.B16, V31.B16, V31.B16
	VFCMEQ_D2(31, 31, 30)

	LSR $1, R2, R4
	CBZ R4, selu_neon_done

selu_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R3), [V1.D2]
	VFMUL_D2(20, 0, 2)                         // positive: scale*x
	VFSUB_D2(22, 1, 3)                         // exp(x)-1
	VFMUL_D2(21, 3, 3)                         // negative branch
	VFCMGT_D2(31, 0, 4)                        // x > 0
	VAND V4.B16, V2.B16, V2.B16
	VEOR V4.B16, V30.B16, V5.B16
	VAND V5.B16, V3.B16, V3.B16
	VORR  V3.B16, V2.B16, V2.B16
	VST1.P [V2.D2], 16(R0)
	SUBS $1, R4, R4
	BNE  selu_neon_loop

selu_neon_done:
	RET
