#include "textflag.h"

DATA ·seluScale+0(SB)/8, $1.0507009873554805
GLOBL ·seluScale(SB), RODATA, $8
DATA ·seluScaleAlpha+0(SB)/8, $1.7580993408473766
GLOBL ·seluScaleAlpha(SB), RODATA, $8

// seluBlendNEON(dst, src, expValues []float64)
TEXT ·seluBlendNEON(SB), NOSPLIT, $0-72
	MOVD dst+0(FP), R0
	MOVD src_base+24(FP), R1
	MOVD src_len+32(FP), R2
	MOVD expValues_base+48(FP), R3

	FMOVD ·seluScale(SB), F20
	FMOVD ·seluScaleAlpha(SB), F21
	FMOVD $1.0, F22
	FMOVD $0.0, F23

	LSR $1, R2, R4
	CBZ R4, selu_neon_done

selu_neon_pair:
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R1), F1
	FMOVD.P 8(R3), F2
	FMOVD.P 8(R3), F3

	FCMPD F23, F0
	BGT   selu_neon_pos0
	FSUBD F22, F2, F4
	FMULD F21, F4, F4
	JMP   selu_neon_lane1
selu_neon_pos0:
	FMULD F20, F0, F4

selu_neon_lane1:
	FCMPD F23, F1
	BGT   selu_neon_pos1
	FSUBD F22, F3, F5
	FMULD F21, F5, F5
	JMP   selu_neon_store
selu_neon_pos1:
	FMULD F20, F1, F5

selu_neon_store:
	FMOVD.P F4, 8(R0)
	FMOVD.P F5, 8(R0)
	SUBS $1, R4, R4
	BNE  selu_neon_pair

selu_neon_done:
	RET
