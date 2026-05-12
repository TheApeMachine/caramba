#include "textflag.h"

// adamStepNEON(out, mState, vState, params, grads []float64,
//              beta1, oneMinusBeta1, beta2, oneMinusBeta2,
//              lrT, eps, lrWD float64)
//
// Fused per-element Adam update. ARM64 scalar-FP (Plan-9 toolchain offers
// limited NEON .2D mnemonics; double-precision FMA via FMADDD/FMSUBD).
// Two interleaved chains per loop iteration for ILP.
//
// ABI0:
//   out+0..23
//   m+24..47
//   v+48..71
//   params+72..95
//   grads+96..119
//   beta1+120, oneMinusBeta1+128, beta2+136, oneMinusBeta2+144,
//   lrT+152, eps+160, lrWD+168
//
// Plan-9 operand semantics:
//   FMADDD A, B, C, D  →  D = A*C + B
//   FMSUBD A, B, C, D  →  D = B - A*C

TEXT ·adamStepNEON(SB), NOSPLIT, $0-176
	MOVD out+0(FP), R0
	MOVD m+24(FP), R1
	MOVD v+48(FP), R2
	MOVD params+72(FP), R3
	MOVD grads+96(FP), R4
	MOVD out_len+8(FP), R5

	FMOVD beta1+120(FP), F20
	FMOVD oneMinusBeta1+128(FP), F21
	FMOVD beta2+136(FP), F22
	FMOVD oneMinusBeta2+144(FP), F23
	FMOVD lrT+152(FP), F24
	FMOVD eps+160(FP), F25
	FMOVD lrWD+168(FP), F26

	LSR  $1, R5, R6
	CBZ  R6, adam_neon_tail
adam_neon_loop:
	FMOVD (R1), F0
	FMOVD 8(R1), F1
	FMOVD (R2), F2
	FMOVD 8(R2), F3
	FMOVD (R3), F4
	FMOVD 8(R3), F5
	FMOVD (R4), F6
	FMOVD 8(R4), F7

	// g2 = g*g
	FMULD F6, F6, F8
	FMULD F7, F7, F9

	// m = β1*m + (1-β1)*g     →  m_new = m_old*β1 + g*(1-β1)
	// Step A: m_new = m_old*β1
	FMULD F20, F0, F0
	FMULD F20, F1, F1
	// Step B: m_new += g*(1-β1)   FMADDD A, B, C, D : D = A*C + B
	FMADDD F6, F0, F21, F0
	FMADDD F7, F1, F21, F1
	FMOVD F0, (R1)
	FMOVD F1, 8(R1)

	// v = β2*v + (1-β2)*g²
	FMULD F22, F2, F2
	FMULD F22, F3, F3
	FMADDD F8, F2, F23, F2
	FMADDD F9, F3, F23, F3
	FMOVD F2, (R2)
	FMOVD F3, 8(R2)

	// denom = sqrt(v) + eps
	FSQRTD F2, F10
	FSQRTD F3, F11
	FADDD F25, F10, F10
	FADDD F25, F11, F11

	// upd = m / denom
	FDIVD F10, F0, F12
	FDIVD F11, F1, F13

	// out = params*(1-lrWD) - lrT*upd
	// tmp = params - lrWD*params   (FMSUBD A, B, C, D : D = B - A*C)
	FMSUBD F26, F4, F4, F4
	FMSUBD F26, F5, F5, F5
	// out = tmp - lrT*upd
	FMSUBD F24, F4, F12, F4
	FMSUBD F24, F5, F13, F5

	FMOVD F4, (R0)
	FMOVD F5, 8(R0)

	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	ADD $16, R3, R3
	ADD $16, R4, R4
	SUBS $1, R6, R6
	BNE  adam_neon_loop

adam_neon_tail:
	AND $1, R5, R7
	CBZ R7, adam_neon_done

	FMOVD (R1), F0
	FMOVD (R2), F2
	FMOVD (R3), F4
	FMOVD (R4), F6

	FMULD F6, F6, F8

	FMULD F20, F0, F0
	FMADDD F6, F0, F21, F0
	FMOVD F0, (R1)

	FMULD F22, F2, F2
	FMADDD F8, F2, F23, F2
	FMOVD F2, (R2)

	FSQRTD F2, F10
	FADDD F25, F10, F10

	FDIVD F10, F0, F12

	FMSUBD F26, F4, F4, F4
	FMSUBD F24, F4, F12, F4

	FMOVD F4, (R0)

adam_neon_done:
	RET
