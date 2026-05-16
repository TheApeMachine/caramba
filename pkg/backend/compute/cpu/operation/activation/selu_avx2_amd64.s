#include "textflag.h"

DATA ·seluScale+0(SB)/8, $1.0507009873554805
GLOBL ·seluScale(SB), RODATA, $8
DATA ·seluScaleAlpha+0(SB)/8, $1.7580993408473766
GLOBL ·seluScaleAlpha(SB), RODATA, $8

// seluBlendAVX2(dst, src, expValues []float64)
TEXT ·seluBlendAVX2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ src_base+24(FP), DI
	MOVQ src_len+32(FP), BX
	MOVQ expValues_base+48(FP), SI

	VXORPD Y0, Y0, Y0
	VBROADCASTSD ·seluScale(SB), Y10
	VBROADCASTSD ·seluScaleAlpha(SB), Y11
	VBROADCASTSD ·atC0(SB), Y12

	CMPQ BX, $4
	JL   selu_avx2_done

selu_avx2_loop:
	VMOVUPD (DI), Y1
	VMOVUPD (SI), Y2

	VMULPD Y10, Y1, Y3
	VSUBPD Y12, Y2, Y4
	VMULPD Y11, Y4, Y4
	VCMPPD $14, Y0, Y1, Y5
	VBLENDVPD Y5, Y3, Y4, Y6
	VMOVUPD Y6, (AX)

	ADDQ $32, AX
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  selu_avx2_loop

selu_avx2_done:
	VZEROUPPER
	RET
