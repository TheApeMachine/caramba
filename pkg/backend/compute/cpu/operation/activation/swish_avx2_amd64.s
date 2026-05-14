#include "textflag.h"

DATA ·swishConst27_amd64+0(SB)/8, $27.0
GLOBL ·swishConst27_amd64(SB), RODATA, $8
DATA ·swishConst9_amd64+0(SB)/8, $9.0
GLOBL ·swishConst9_amd64(SB), RODATA, $8
DATA ·swishHalf_amd64+0(SB)/8, $0.5
GLOBL ·swishHalf_amd64(SB), RODATA, $8
DATA ·swishOne_amd64+0(SB)/8, $1.0
GLOBL ·swishOne_amd64(SB), RODATA, $8
DATA ·swishNegOne_amd64+0(SB)/8, $-1.0
GLOBL ·swishNegOne_amd64(SB), RODATA, $8

// SwishAVX2(dst, src []float64)
// dst[i] = src[i] * sigmoid(src[i])
TEXT ·SwishAVX2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src_base+24(FP), DI
	MOVQ src_len+32(FP), SI
	CMPQ SI, $0
	JLE  avx2_done

	VMOVSD ·swishConst27_amd64(SB), X10
	VBROADCASTSD X10, Y10
	VMOVSD ·swishConst9_amd64(SB), X11
	VBROADCASTSD X11, Y11
	VMOVSD ·swishHalf_amd64(SB), X12
	VBROADCASTSD X12, Y12
	VMOVSD ·swishOne_amd64(SB), X13
	VBROADCASTSD X13, Y13
	VMOVSD ·swishNegOne_amd64(SB), X14
	VBROADCASTSD X14, Y14

avx2_loop:
	VMOVUPD (DI), Y0
	VMULPD Y12, Y0, Y1
	VMULPD Y1, Y1, Y2
	VADDPD Y10, Y2, Y3
	VMULPD Y11, Y2, Y4
	VADDPD Y10, Y4, Y4
	VMULPD Y1, Y3, Y5
	VDIVPD Y4, Y5, Y6
	VMINPD Y13, Y6, Y6
	VMAXPD Y14, Y6, Y6
	VADDPD Y13, Y6, Y6
	VMULPD Y12, Y6, Y6
	VMULPD Y0, Y6, Y6
	VMOVUPD Y6, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, SI
	CMPQ SI, $4
	JGE  avx2_loop

avx2_done:
	VZEROUPPER
	RET

// swishScalarAMD64(dst, src []float64)
TEXT ·swishScalarAMD64(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src_base+24(FP), DI
	MOVQ src_len+32(FP), SI
	CMPQ SI, $0
	JLE  scalar_done

	MOVSD ·swishConst27_amd64(SB), X10
	MOVSD ·swishConst9_amd64(SB), X11
	MOVSD ·swishHalf_amd64(SB), X12
	MOVSD ·swishOne_amd64(SB), X13
	MOVSD ·swishNegOne_amd64(SB), X14

scalar_loop:
	MOVSD (DI), X0
	MOVAPD X0, X1
	MULSD  X12, X1
	MOVAPD X1, X2
	MULSD  X1, X2
	MOVAPD X2, X3
	ADDSD  X10, X3
	MOVAPD X2, X4
	MULSD  X11, X4
	ADDSD  X10, X4
	MOVAPD X1, X5
	MULSD  X3, X5
	MOVAPD X5, X6
	DIVSD  X4, X6
	MINSD  X13, X6
	MAXSD  X14, X6
	ADDSD  X13, X6
	MULSD  X12, X6
	MULSD  X0, X6
	MOVSD  X6, (AX)
	ADDQ $8, AX
	ADDQ $8, DI
	SUBQ $1, SI
	JG   scalar_loop

scalar_done:
	RET
