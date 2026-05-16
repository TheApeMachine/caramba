#include "textflag.h"

// seluBlendSSE2(dst, src, expValues []float64)
TEXT ·seluBlendSSE2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ src_base+24(FP), DI
	MOVQ src_len+32(FP), BX
	MOVQ expValues_base+48(FP), SI

	PXOR   X0, X0
	MOVSD  ·seluScale(SB), X10
	SHUFPD $0, X10, X10
	MOVSD  ·seluScaleAlpha(SB), X11
	SHUFPD $0, X11, X11
	MOVSD  ·atC0(SB), X12
	SHUFPD $0, X12, X12

	CMPQ BX, $2
	JL   selu_sse2_done

selu_sse2_loop:
	MOVUPD (DI), X1
	MOVUPD (SI), X2

	MOVAPD X1, X3
	MULPD  X10, X3
	SUBPD  X12, X2
	MULPD  X11, X2

	MOVAPD X1, X4
	CMPPD  X0, X4, $14
	MOVAPD X4, X5
	ANDPD  X3, X5
	MOVAPD X4, X6
	ANDNPD X2, X6
	ORPD   X6, X5
	MOVUPD X5, (AX)

	ADDQ $16, AX
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  selu_sse2_loop

selu_sse2_done:
	RET
