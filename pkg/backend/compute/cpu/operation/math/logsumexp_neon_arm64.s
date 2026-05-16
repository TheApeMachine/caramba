#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFMINNM_D2(m, n, d) WORD $(0x4EE0C400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMAXNM_D2(m, n, d) WORD $(0x4E60C400 | ((m) << 16) | ((n) << 5) | (d))
#define VFRINTN_D2(n, d) WORD $(0x4E618800 | ((n) << 5) | (d))
#define VFCVTZS_D2(n, d) WORD $(0x4EE1B800 | ((n) << 5) | (d))
#define VLOADDUP(sym, addr, vec) MOVD $sym, addr; VLD1R (addr), [vec.D2]

// logSumExpRowNEON(row []float64) float64
TEXT ·logSumExpRowNEON(SB), NOSPLIT, $32-32
	MOVD row+0(FP), R0
	MOVD row_len+8(FP), R1
	CBZ R1, lsen_done_zero
	MOVD R1, R11
	MOVD R0, R12

	CMP $2, R1
	BLT lsen_max_single
	MOVD R0, R7
	LSR $1, R1, R8
	VLD1.P 16(R7), [V0.D2]
	SUBS $1, R8, R8
	BEQ lsen_max_horizontal
lsen_max_loop:
	VLD1.P 16(R7), [V1.D2]
	VFMAXNM_D2(1, 0, 0)
	SUBS $1, R8, R8
	BNE lsen_max_loop
lsen_max_horizontal:
	MOVD RSP, R8
	VST1.P [V0.D2], 16(R8)
	FMOVD 0(RSP), F10
	FMOVD 8(RSP), F1
	FMAXD F1, F10, F10
	TST $1, R11
	BEQ lsen_have_max
	FMOVD (R7), F1
	FMAXD F1, F10, F10
	B lsen_have_max
lsen_max_single:
	FMOVD (R0), F10

lsen_have_max:
	FMOVD F10, 0(RSP)
	VLD1R (RSP), [V10.D2]
	VEOR V11.B16, V11.B16, V11.B16
	VLOADDUP(·expLog2E(SB), R9, V20)
	VLOADDUP(·expLn2Hi(SB), R9, V21)
	VLOADDUP(·expLn2Lo(SB), R9, V22)
	VLOADDUP(·expMaxArg(SB), R9, V23)
	VLOADDUP(·expMinArg(SB), R9, V24)
	VLOADDUP(·expC0(SB), R9, V25)
	MOVD $1023, R10
	VDUP R10, V27.D2

	MOVD R12, R7
	LSR $1, R11, R8
	CBZ R8, lsen_exp_tail
lsen_exp_loop:
	VLD1.P 16(R7), [V0.D2]
	VFSUB_D2(10, 0, 0)
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
	VFADD_D2(2, 11, 11)
	SUBS $1, R8, R8
	BNE lsen_exp_loop

lsen_exp_tail:
	MOVD RSP, R8
	VST1.P [V11.D2], 16(R8)
	FMOVD 0(RSP), F14
	FMOVD 8(RSP), F8
	FADDD F8, F14, F14
	TST $1, R11
	BEQ lsen_have_sum

	FMOVD (R7), F2
	FSUBD F10, F2, F2
	FMINNMD F23, F2, F2
	FMAXNMD F24, F2, F2
	FMULD F20, F2, F3
	FRINTND F3, F3
	FMSUBD F3, F2, F21, F2
	FMSUBD F3, F2, F22, F2
	FMOVD ·expC11(SB), F4
	FMOVD ·expC10(SB), F5
	FMADDD F4, F5, F2, F4
	FMOVD ·expC9(SB), F5
	FMADDD F4, F5, F2, F4
	FMOVD ·expC8(SB), F5
	FMADDD F4, F5, F2, F4
	FMOVD ·expC7(SB), F5
	FMADDD F4, F5, F2, F4
	FMOVD ·expC6(SB), F5
	FMADDD F4, F5, F2, F4
	FMOVD ·expC5(SB), F5
	FMADDD F4, F5, F2, F4
	FMOVD ·expC4(SB), F5
	FMADDD F4, F5, F2, F4
	FMOVD ·expC3(SB), F5
	FMADDD F4, F5, F2, F4
	FMOVD ·expC2(SB), F5
	FMADDD F4, F5, F2, F4
	FMOVD ·expC1(SB), F5
	FMADDD F4, F5, F2, F4
	FMOVD ·expC0(SB), F5
	FMADDD F4, F5, F2, F4
	FCVTZSD F3, R5
	ADD R10, R5, R5
	LSL $52, R5, R5
	FMOVD R5, F6
	FMULD F6, F4, F4
	FADDD F4, F14, F14

lsen_have_sum:
	FMOVD F14, R5
	LSR $52, R5, R6
	AND $0x7FF, R6, R6
	SUB R10, R6, R6
	SCVTFD R6, F8

	MOVD $0x000FFFFFFFFFFFFF, R7
	AND R7, R5, R7
	MOVD $0x3FF0000000000000, R6
	ORR R6, R7, R7
	FMOVD R7, F9

	FMOVD ·logSqrt2(SB), F1
	FCMPD F1, F9
	BLE lsen_no_shift
	FMOVD ·logHalf(SB), F2
	FMULD F2, F9, F9
	FMOVD $1.0, F2
	FADDD F2, F8, F8
lsen_no_shift:
	FMOVD $1.0, F1
	FSUBD F1, F9, F2
	FADDD F1, F9, F3
	FDIVD F3, F2, F2
	FMULD F2, F2, F3

	FMOVD ·logA6(SB), F4
	FMOVD ·logA5(SB), F5
	FMADDD F4, F5, F3, F4
	FMOVD ·logA4(SB), F5
	FMADDD F4, F5, F3, F4
	FMOVD ·logA3(SB), F5
	FMADDD F4, F5, F3, F4
	FMOVD ·logA2(SB), F5
	FMADDD F4, F5, F3, F4
	FMOVD ·logA1(SB), F5
	FMADDD F4, F5, F3, F4
	FMOVD ·logA0(SB), F5
	FMADDD F4, F5, F3, F4

	FMULD F2, F4, F4
	FADDD F4, F4, F4
	FMOVD ·logLn2(SB), F5
	FMADDD F5, F4, F8, F4
	FADDD F10, F4, F4
	FMOVD F4, ret+24(FP)
	RET

lsen_done_zero:
	FMOVD $0.0, F0
	FMOVD F0, ret+24(FP)
	RET
