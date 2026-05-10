#include "textflag.h"

DATA ·geluConstNeon27+0(SB)/8, $27.0
GLOBL ·geluConstNeon27(SB), RODATA|NOPTR, $8
DATA ·geluConstNeon9+0(SB)/8, $9.0
GLOBL ·geluConstNeon9(SB), RODATA|NOPTR, $8
DATA ·geluHalfNeon+0(SB)/8, $0.5
GLOBL ·geluHalfNeon(SB), RODATA|NOPTR, $8
DATA ·geluOneNeon+0(SB)/8, $1.0
GLOBL ·geluOneNeon(SB), RODATA|NOPTR, $8
DATA ·geluC1Neon+0(SB)/8, $0.79788456080286535587989211986876
GLOBL ·geluC1Neon(SB), RODATA|NOPTR, $8
DATA ·geluC2Neon+0(SB)/8, $0.044715
GLOBL ·geluC2Neon(SB), RODATA|NOPTR, $8
DATA ·geluClampPos+0(SB)/8, $5.0
GLOBL ·geluClampPos(SB), RODATA|NOPTR, $8
DATA ·geluClampNeg+0(SB)/8, $-5.0
GLOBL ·geluClampNeg(SB), RODATA|NOPTR, $8

// GeLUNEON(dst, x []float64)
// ABI0: dst+0(FP) ptr/len/cap, x+24(FP) ptr/len/cap
// Computes 0.5*x*(1+tanh(c1*(x+c2*x^3))).
// Uses rational approx tanh(z)≈z*(27+z^2)/(27+9*z^2) clamped to z∈[-5,5].
TEXT ·GeLUNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP),   R0
	MOVD x_base+24(FP), R3
	MOVD x_len+32(FP), R4

	FMOVD ·geluC2Neon(SB),    F20
	FMOVD ·geluC1Neon(SB),    F21
	FMOVD ·geluConstNeon27(SB), F22
	FMOVD ·geluConstNeon9(SB), F23
	FMOVD ·geluOneNeon(SB),   F24
	FMOVD ·geluHalfNeon(SB),  F25
	FMOVD ·geluClampPos(SB),  F26   // +5.0
	FMOVD ·geluClampNeg(SB),  F27   // -5.0

	CBZ R4, gelu_done

gelu_loop:
	FMOVD.P 8(R3), F0       // x

	// z = c1 * (x + c2*x^3)
	FMULD F0, F0, F1         // x^2
	FMULD F1, F0, F2         // x^3
	FMULD F20, F2, F3        // c2*x^3
	FADDD F3, F0, F3         // x + c2*x^3
	FMULD F21, F3, F3        // z = c1*(x+c2*x^3)

	// clamp z to [-5, 5]
	FMIND F26, F3, F3        // z = min(z, 5)
	FMAXD F27, F3, F3        // z = max(z, -5)

	// rational tanh: z*(27+z^2)/(27+9*z^2)
	FMULD F3, F3, F4         // z^2
	FADDD F22, F4, F5        // 27+z^2
	FMULD F23, F4, F6        // 9*z^2
	FADDD F22, F6, F6        // 27+9*z^2
	FMULD F3, F5, F7         // z*(27+z^2)
	FDIVD F6, F7, F8         // tanh(z) approx (may exceed ±1 near clamp)

	// clamp tanh output to [-1, 1]
	FMIND F26, F8, F8        // min(tanh, 5) → reuse F26=5? No, use F24=1
	// F24=1.0, F27=-5.0 → need constants 1 and -1
	// Use: min(F8, F24) where F24=1.0, then max with negated F24
	FMIND F24, F8, F8        // min(tanh, 1)
	FNEGD F24, F28           // F28 = -1.0
	FMAXD F28, F8, F8        // max(tanh, -1)

	// out = 0.5*x*(1+tanh)
	FADDD F24, F8, F8        // 1+tanh
	FMULD F0, F8, F8         // x*(1+tanh)
	FMULD F25, F8, F8        // 0.5*x*(1+tanh)
	FMOVD.P F8, 8(R0)

	SUBS $1, R4, R4
	BNE  gelu_loop

gelu_done:
	RET
