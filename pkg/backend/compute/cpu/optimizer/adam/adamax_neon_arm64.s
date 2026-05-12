#include "textflag.h"

DATA ·adamaxAbsMask+0(SB)/8, $0x7FFFFFFFFFFFFFFF
GLOBL ·adamaxAbsMask(SB), RODATA, $8

// adamaxStepNEON(out, m, u, params, grads []float64,
//                beta1, oneMinusBeta1, beta2, lrT, eps float64)
// Plan-9 ARM64 scalar-FP, two interleaved chains.
TEXT ·adamaxStepNEON(SB), NOSPLIT, $0-160
	MOVD out+0(FP), R0
	MOVD m+24(FP), R1
	MOVD u+48(FP), R2
	MOVD params+72(FP), R3
	MOVD grads+96(FP), R4
	MOVD out_len+8(FP), R5

	FMOVD beta1+120(FP), F20
	FMOVD oneMinusBeta1+128(FP), F21
	FMOVD beta2+136(FP), F22
	FMOVD lrT+144(FP), F23
	FMOVD eps+152(FP), F24
	MOVD  ·adamaxAbsMask(SB), R10
	FMOVD R10, F25

	LSR  $1, R5, R6
	CBZ  R6, adamax_neon_tail
adamax_neon_loop:
	FMOVD (R1), F0
	FMOVD 8(R1), F1
	FMOVD (R2), F2
	FMOVD 8(R2), F3
	FMOVD (R3), F4
	FMOVD 8(R3), F5
	FMOVD (R4), F6
	FMOVD 8(R4), F7

	// m = β1*m + (1-β1)*g
	FMULD F20, F0, F0
	FMULD F20, F1, F1
	FMADDD F6, F0, F21, F0
	FMADDD F7, F1, F21, F1
	FMOVD F0, (R1)
	FMOVD F1, 8(R1)

	// |g|
	FMOVD F6, R11
	AND   R10, R11, R11
	FMOVD R11, F8
	FMOVD F7, R11
	AND   R10, R11, R11
	FMOVD R11, F9

	// u = max(β2*u, |g|)
	FMULD F22, F2, F2
	FMULD F22, F3, F3
	FMAXD F8, F2, F2
	FMAXD F9, F3, F3
	FMOVD F2, (R2)
	FMOVD F3, 8(R2)

	// denom = u + eps
	FADDD F24, F2, F10
	FADDD F24, F3, F11
	// upd = m / denom
	FDIVD F10, F0, F12
	FDIVD F11, F1, F13
	// out = params - lrT*upd
	FMSUBD F23, F4, F12, F4
	FMSUBD F23, F5, F13, F5
	FMOVD F4, (R0)
	FMOVD F5, 8(R0)

	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	ADD $16, R3, R3
	ADD $16, R4, R4
	SUBS $1, R6, R6
	BNE  adamax_neon_loop

adamax_neon_tail:
	AND $1, R5, R7
	CBZ R7, adamax_neon_done

	FMOVD (R1), F0
	FMOVD (R2), F2
	FMOVD (R3), F4
	FMOVD (R4), F6

	FMULD F20, F0, F0
	FMADDD F6, F0, F21, F0
	FMOVD F0, (R1)

	FMOVD F6, R11
	AND R10, R11, R11
	FMOVD R11, F8

	FMULD F22, F2, F2
	FMAXD F8, F2, F2
	FMOVD F2, (R2)

	FADDD F24, F2, F10
	FDIVD F10, F0, F12
	FMSUBD F23, F4, F12, F4
	FMOVD F4, (R0)

adamax_neon_done:
	RET
