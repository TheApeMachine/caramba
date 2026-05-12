#include "textflag.h"

DATA ·logOne+0(SB)/8, $1.0
GLOBL ·logOne(SB), RODATA, $8
DATA ·logHalf+0(SB)/8, $0.5
GLOBL ·logHalf(SB), RODATA, $8
DATA ·logSqrt2+0(SB)/8, $1.4142135623730951
GLOBL ·logSqrt2(SB), RODATA, $8
DATA ·logLn2+0(SB)/8, $0.6931471805599453
GLOBL ·logLn2(SB), RODATA, $8
DATA ·logA0+0(SB)/8, $1.0
GLOBL ·logA0(SB), RODATA, $8
DATA ·logA1+0(SB)/8, $0.3333333333333333
GLOBL ·logA1(SB), RODATA, $8
DATA ·logA2+0(SB)/8, $0.2
GLOBL ·logA2(SB), RODATA, $8
DATA ·logA3+0(SB)/8, $0.14285714285714285
GLOBL ·logA3(SB), RODATA, $8
DATA ·logA4+0(SB)/8, $0.1111111111111111
GLOBL ·logA4(SB), RODATA, $8
DATA ·logA5+0(SB)/8, $0.09090909090909091
GLOBL ·logA5(SB), RODATA, $8
DATA ·logA6+0(SB)/8, $0.07692307692307693
GLOBL ·logA6(SB), RODATA, $8

// logVecNEON(dst, src []float64)
TEXT ·logVecNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD src_len+32(FP), R2

	FMOVD ·logOne(SB), F10
	FMOVD ·logHalf(SB), F11
	FMOVD ·logSqrt2(SB), F12
	FMOVD ·logLn2(SB), F13
	FMOVD ·logA0(SB), F14
	FMOVD ·logA1(SB), F15
	FMOVD ·logA2(SB), F16
	FMOVD ·logA3(SB), F17
	FMOVD ·logA4(SB), F18
	FMOVD ·logA5(SB), F19
	FMOVD ·logA6(SB), F20

	MOVD $0x000FFFFFFFFFFFFF, R8
	MOVD $0x3FF0000000000000, R9
	MOVD $1023, R10
	MOVD $0x7FF, R11

	CBZ R2, done_log_neon

loop_log_neon:
	FMOVD.P 8(R1), F0                   // x
	FMOVD   F0, R3                      // bits

	LSR $52, R3, R4
	AND R11, R4, R4
	SUB R10, R4, R4                     // R4 = raw_exp - 1023 (signed int64)

	AND R8, R3, R5
	ORR R9, R5, R5
	FMOVD R5, F1                        // m

	FCMPD F12, F1
	BLE skip_norm
	FMULD F11, F1, F1                   // m *= 0.5
	ADD $1, R4, R4
skip_norm:

	FSUBD F10, F1, F3                   // t numerator = m - 1
	FADDD F10, F1, F4                   // denom = m + 1
	FDIVD F4, F3, F3                    // t
	FMULD F3, F3, F4                    // t^2

	// Horner Pp = a6 + u*(a5 + u*(... + u*a0))
	FMOVD F20, F5                       // P = a6
	FMADDD F5, F19, F4, F5              // P = P*u + a5
	FMADDD F5, F18, F4, F5              // P = P*u + a4
	FMADDD F5, F17, F4, F5
	FMADDD F5, F16, F4, F5
	FMADDD F5, F15, F4, F5
	FMADDD F5, F14, F4, F5              // P = P*u + a0

	FMULD F3, F5, F5                    // t*P
	FADDD F5, F5, F5                    // 2*t*P = log(m)

	SCVTFD R4, F6                       // e as double
	FMADDD F6, F5, F13, F5              // F5 = F6*F13 + F5 = e*ln2 + log(m)

	FMOVD.P F5, 8(R0)
	SUBS $1, R2, R2
	BNE  loop_log_neon

done_log_neon:
	RET
