#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFMINNM_D2(m, n, d) WORD $(0x4EE0C400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMAXNM_D2(m, n, d) WORD $(0x4E60C400 | ((m) << 16) | ((n) << 5) | (d))
#define VFRINTN_D2(n, d) WORD $(0x4E618800 | ((n) << 5) | (d))
#define VFCVTZS_D2(n, d) WORD $(0x4EE1B800 | ((n) << 5) | (d))
#define VLOADDUP(sym, addr, vec) MOVD $sym, addr; VLD1R (addr), [vec.D2]

// Constants identical to amd64 path.
DATA ·expLog2E+0(SB)/8, $1.4426950408889634
GLOBL ·expLog2E(SB), RODATA, $8
DATA ·expLn2Hi+0(SB)/8, $0.6931471803691864
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

// expVecNEON(dst, src []float64)
// Two-lane NEON implementation with scalar odd-tail handling.
TEXT ·expVecNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD src_len+32(FP), R2

	VLOADDUP(·expLog2E(SB), R9, V20)
	VLOADDUP(·expLn2Hi(SB), R9, V21)
	VLOADDUP(·expLn2Lo(SB), R9, V22)
	VLOADDUP(·expMaxArg(SB), R9, V23)
	VLOADDUP(·expMinArg(SB), R9, V24)
	VLOADDUP(·expC0(SB), R9, V25)
	MOVD  $1023, R7
	VDUP R7, V27.D2

	LSR  $1, R2, R3
	CBZ  R3, done_exp_neon

loop_exp_neon:
	VLD1.P 16(R1), [V0.D2]
	VFMINNM_D2(23, 0, 0)
	VFMAXNM_D2(24, 0, 0)

	VFMUL_D2(20, 0, 1)
	VFRINTN_D2(1, 1)
	VFMUL_D2(21, 1, 3)
	VFSUB_D2(3, 0, 0)
	VFMUL_D2(22, 1, 3)
	VFSUB_D2(3, 0, 0)

	VLOADDUP(·expC11(SB), R9, V2)
	VLOADDUP(·expC10(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·expC9(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·expC8(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·expC7(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·expC6(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·expC5(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·expC4(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·expC3(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·expC2(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·expC1(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(25, 2, 2)

	VFCVTZS_D2(1, 1)
	VADD V27.D2, V1.D2, V1.D2
	VSHL $52, V1.D2, V1.D2
	VFMUL_D2(1, 2, 2)
	VST1.P [V2.D2], 16(R0)

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
	FMOVD   ·expC7(SB), F3
	FMADDD  F2, F3, F0, F2
	FMOVD   ·expC6(SB), F3
	FMADDD  F2, F3, F0, F2
	FMOVD   ·expC5(SB), F3
	FMADDD  F2, F3, F0, F2
	FMOVD   ·expC4(SB), F3
	FMADDD  F2, F3, F0, F2
	FMOVD   ·expC3(SB), F3
	FMADDD  F2, F3, F0, F2
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
