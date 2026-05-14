#include "textflag.h"

DATA ·swishConst27+0(SB)/8, $27.0
GLOBL ·swishConst27(SB), RODATA|NOPTR, $8
DATA ·swishConst9+0(SB)/8, $9.0
GLOBL ·swishConst9(SB), RODATA|NOPTR, $8
DATA ·swishHalf+0(SB)/8, $0.5
GLOBL ·swishHalf(SB), RODATA|NOPTR, $8
DATA ·swishOne+0(SB)/8, $1.0
GLOBL ·swishOne(SB), RODATA|NOPTR, $8
DATA ·swishNegOne+0(SB)/8, $-1.0
GLOBL ·swishNegOne(SB), RODATA|NOPTR, $8

// swishNEON(dst, src []float64)
// dst[i] = src[i] * sigmoid(src[i]) using the local fast sigmoid form:
//   t = x / 2
//   sigmoid(x) ≈ (1 + clamp(-1, 1, t * (27 + t*t) / (27 + 9*t*t))) / 2
//
// The constants 27 and 9 implement a bounded rational tanh-style
// approximation; no external citation is assumed. The clamp instructions below
// bound the rational result to [-1, 1]. The denominator 27 + 9*t*t is always at
// least 27 for finite t, so the division has no zero or underflow denominator
// in the supported finite input range. Tests cover max Swish error across
// [-12, 12] against the exact x/(1+exp(-x)) definition.
TEXT ·swishNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src_base+24(FP), R1
	MOVD src_len+32(FP), R2
	CBZ  R2, done

	FMOVD ·swishConst27(SB), F26
	FMOVD ·swishConst9(SB), F27
	FMOVD ·swishHalf(SB), F28
	FMOVD ·swishOne(SB), F29
	FMOVD ·swishNegOne(SB), F30

	LSR $1, R2, R3
	CBZ R3, scalar_tail

pair_loop:
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R1), F10

	FMULD F28, F0, F1
	FMULD F1, F1, F2
	FADDD F26, F2, F3
	FMULD F27, F2, F4
	FADDD F26, F4, F4
	FMULD F1, F3, F5
	FDIVD F4, F5, F6
	FMIND F29, F6, F6
	FMAXD F30, F6, F6
	FADDD F29, F6, F6
	FMULD F28, F6, F6
	FMULD F0, F6, F6

	FMULD F28, F10, F11
	FMULD F11, F11, F12
	FADDD F26, F12, F13
	FMULD F27, F12, F14
	FADDD F26, F14, F14
	FMULD F11, F13, F15
	FDIVD F14, F15, F16
	FMIND F29, F16, F16
	FMAXD F30, F16, F16
	FADDD F29, F16, F16
	FMULD F28, F16, F16
	FMULD F10, F16, F16

	FMOVD.P F6, 8(R0)
	FMOVD.P F16, 8(R0)
	SUBS $1, R3, R3
	BNE  pair_loop

	TBZ $0, R2, done

scalar_tail:
	FMOVD (R1), F0
	FMULD F28, F0, F1
	FMULD F1, F1, F2
	FADDD F26, F2, F3
	FMULD F27, F2, F4
	FADDD F26, F4, F4
	FMULD F1, F3, F5
	FDIVD F4, F5, F6
	FMIND F29, F6, F6
	FMAXD F30, F6, F6
	FADDD F29, F6, F6
	FMULD F28, F6, F6
	FMULD F0, F6, F6
	FMOVD F6, (R0)

done:
	RET
