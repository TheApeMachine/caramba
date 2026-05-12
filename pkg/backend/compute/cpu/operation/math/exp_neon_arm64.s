#include "textflag.h"

// Constants identical to amd64 path.
DATA ·expLog2E+0(SB)/8, $1.4426950408889634
GLOBL ·expLog2E(SB), RODATA, $8
DATA ·expLn2Hi+0(SB)/8, $0.6931471805599453
GLOBL ·expLn2Hi(SB), RODATA, $8
DATA ·expLn2Lo+0(SB)/8, $1.9082149292705877e-10
GLOBL ·expLn2Lo(SB), RODATA, $8
DATA ·expMaxArg+0(SB)/8, $709.0
GLOBL ·expMaxArg(SB), RODATA, $8
DATA ·expMinArg+0(SB)/8, $-708.0
GLOBL ·expMinArg(SB), RODATA, $8

DATA ·expC0+0(SB)/8, $1.0
GLOBL ·expC0(SB), RODATA, $8
DATA ·expC1+0(SB)/8, $1.0
GLOBL ·expC1(SB), RODATA, $8
DATA ·expC2+0(SB)/8, $0.5
GLOBL ·expC2(SB), RODATA, $8
DATA ·expC3+0(SB)/8, $0.16666666666666666
GLOBL ·expC3(SB), RODATA, $8
DATA ·expC4+0(SB)/8, $0.041666666666666664
GLOBL ·expC4(SB), RODATA, $8
DATA ·expC5+0(SB)/8, $0.008333333333333333
GLOBL ·expC5(SB), RODATA, $8
DATA ·expC6+0(SB)/8, $0.001388888888888889
GLOBL ·expC6(SB), RODATA, $8
DATA ·expC7+0(SB)/8, $0.0001984126984126984
GLOBL ·expC7(SB), RODATA, $8
DATA ·expC8+0(SB)/8, $2.4801587301587302e-5
GLOBL ·expC8(SB), RODATA, $8
DATA ·expC9+0(SB)/8, $2.7557319223985893e-6
GLOBL ·expC9(SB), RODATA, $8
DATA ·expC10+0(SB)/8, $2.7557319223985894e-7
GLOBL ·expC10(SB), RODATA, $8
DATA ·expC11+0(SB)/8, $2.5052108385441718e-8
GLOBL ·expC11(SB), RODATA, $8

// Plan 9 ARM64 operand semantics:
//   FMADDD A, B, C, D  →  D = A*C + B
//   FMSUBD A, B, C, D  →  D = B - A*C
//
// expVecNEON(dst, src []float64)
// Scalar-FP ARM64 implementation. Two interleaved chains for ILP.
TEXT ·expVecNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD src_len+32(FP), R2

	FMOVD ·expLog2E(SB), F20
	FMOVD ·expLn2Hi(SB), F21
	FMOVD ·expLn2Lo(SB), F22
	FMOVD ·expMaxArg(SB), F23
	FMOVD ·expMinArg(SB), F24
	FMOVD ·expC7(SB), F25
	FMOVD ·expC6(SB), F26
	FMOVD ·expC5(SB), F27
	FMOVD ·expC4(SB), F28
	FMOVD ·expC3(SB), F29
	MOVD  $1023, R7

	LSR  $1, R2, R3
	CBZ  R3, done_exp_neon

loop_exp_neon:
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R1), F10

	FMINNMD F23, F0, F0
	FMAXNMD F24, F0, F0
	FMINNMD F23, F10, F10
	FMAXNMD F24, F10, F10

	FMULD   F20, F0, F1                   // r = x*log2e
	FMULD   F20, F10, F11
	FRINTND F1, F1                        // r = round(r)
	FRINTND F11, F11

	FMSUBD F1, F0, F21, F0                // x = x - r*ln2_hi
	FMSUBD F11, F10, F21, F10
	FMSUBD F1, F0, F22, F0                // x = x - r*ln2_lo  (=f)
	FMSUBD F11, F10, F22, F10

	// Horner: y = y*f + c.   FMADDD y, c, f, y  =>  y = y*f + c
	FMOVD  ·expC11(SB), F2                // y = c11
	FMOVD  ·expC11(SB), F12
	FMOVD  ·expC10(SB), F3
	FMADDD F2, F3, F0, F2                 // y = y*f + c10
	FMADDD F12, F3, F10, F12
	FMOVD  ·expC9(SB), F3
	FMADDD F2, F3, F0, F2
	FMADDD F12, F3, F10, F12
	FMOVD  ·expC8(SB), F3
	FMADDD F2, F3, F0, F2
	FMADDD F12, F3, F10, F12
	FMADDD F2, F25, F0, F2                // y*f + c7
	FMADDD F12, F25, F10, F12
	FMADDD F2, F26, F0, F2                // y*f + c6
	FMADDD F12, F26, F10, F12
	FMADDD F2, F27, F0, F2
	FMADDD F12, F27, F10, F12
	FMADDD F2, F28, F0, F2
	FMADDD F12, F28, F10, F12
	FMADDD F2, F29, F0, F2
	FMADDD F12, F29, F10, F12

	FMOVD  ·expC2(SB), F3
	FMADDD F2, F3, F0, F2
	FMADDD F12, F3, F10, F12
	FMOVD  ·expC1(SB), F3
	FMADDD F2, F3, F0, F2
	FMADDD F12, F3, F10, F12
	FMOVD  ·expC0(SB), F3
	FMADDD F2, F3, F0, F2                 // y = exp(f)
	FMADDD F12, F3, F10, F12

	// 2^r via integer bit-manipulation.
	FCVTZSD F1, R4
	FCVTZSD F11, R5
	ADD R7, R4, R4
	ADD R7, R5, R5
	LSL $52, R4, R4
	LSL $52, R5, R5
	FMOVD R4, F3                          // F3 = 2^r as double
	FMOVD R5, F13

	FMULD F3, F2, F2
	FMULD F13, F12, F12

	FMOVD.P F2, 8(R0)
	FMOVD.P F12, 8(R0)

	SUBS $1, R3, R3
	BNE  loop_exp_neon

done_exp_neon:
	AND $1, R2, R4
	CBZ R4, exit_exp_neon

	FMOVD (R1), F0
	FMINNMD F23, F0, F0
	FMAXNMD F24, F0, F0
	FMULD   F20, F0, F1
	FRINTND F1, F1
	FMSUBD  F1, F0, F21, F0
	FMSUBD  F1, F0, F22, F0
	FMOVD   ·expC11(SB), F2
	FMOVD   ·expC10(SB), F3
	FMADDD  F2, F3, F0, F2
	FMOVD   ·expC9(SB), F3
	FMADDD  F2, F3, F0, F2
	FMOVD   ·expC8(SB), F3
	FMADDD  F2, F3, F0, F2
	FMADDD  F2, F25, F0, F2
	FMADDD  F2, F26, F0, F2
	FMADDD  F2, F27, F0, F2
	FMADDD  F2, F28, F0, F2
	FMADDD  F2, F29, F0, F2
	FMOVD   ·expC2(SB), F3
	FMADDD  F2, F3, F0, F2
	FMOVD   ·expC1(SB), F3
	FMADDD  F2, F3, F0, F2
	FMOVD   ·expC0(SB), F3
	FMADDD  F2, F3, F0, F2
	FCVTZSD F1, R4
	ADD R7, R4, R4
	LSL $52, R4, R4
	FMOVD R4, F3
	FMULD F3, F2, F2
	FMOVD F2, (R0)

exit_exp_neon:
	RET
