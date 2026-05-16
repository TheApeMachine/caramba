#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// choleskyRegNEON(L []float64, n int, eps float64)
TEXT ·choleskyRegNEON(SB), NOSPLIT, $16-40
	MOVD L+0(FP), R0
	MOVD n+24(FP), R1
	FMOVD eps+32(FP), F20
	MOVD $0, R2

cr_col_loop:
	CMP R1, R2
	BGE cr_done

	MOVD R2, R3
	MUL R1, R3
	LSL $3, R3, R3
	MOVD R0, R4
	ADD R3, R4, R4
	MOVD R2, R5
	FMOVD $0.0, F0
	MOVD R4, R6
	CBZ R5, cr_diag
	VEOR V0.B16, V0.B16, V0.B16
	MOVD R5, R11
	LSR  $1, R11, R12
	CBZ  R12, cr_ss_tail
cr_ss_loop:
	VLD1.P 16(R6), [V1.D2]
	VFMUL_D2(1, 1, 2)
	VFADD_D2(2, 0, 0)
	SUBS $1, R12, R12
	BNE  cr_ss_loop

	MOVD RSP, R12
	VST1.P [V0.D2], 16(R12)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0

cr_ss_tail:
	TST $1, R5
	BEQ cr_diag
	FMOVD (R6), F1
	FMADDD F1, F0, F1, F0

cr_diag:
	MOVD R2, R3
	MUL R1, R3
	ADD R2, R3, R3
	LSL $3, R3, R3
	MOVD R0, R6
	ADD R3, R6, R6
	FMOVD (R6), F2
	FSUBD F0, F2, F2
	FMOVD $0.0, F3
	// Plan-9 ARM64: `FCMPD F3, F2` corresponds to ARM `FCMP F2, F3` — flags
	// reflect (F2 - F3). BGT therefore branches when F2 > F3 (pivot > 0),
	// taking the cr_piv_ok path (no clamp). When pivot ≤ 0 or NaN we fall
	// through to the FMOVD F20, F2 clamp.
	FCMPD F3, F2
	BGT cr_piv_ok
	FMOVD F20, F2                            // clamp to eps
cr_piv_ok:
	FSQRTD F2, F2
	FMOVD F2, (R6)
	FMOVD $1.0, F4
	FDIVD F2, F4, F4

	MOVD R2, R7
	ADD $1, R7, R7
cr_row_loop:
	CMP R1, R7
	BGE cr_next_col
	MOVD R7, R3
	MUL R1, R3
	LSL $3, R3, R3
	MOVD R0, R8
	ADD R3, R8, R8
	MOVD R2, R3
	MUL R1, R3
	LSL $3, R3, R3
	MOVD R0, R9
	ADD R3, R9, R9

	MOVD R2, R10
	FMOVD $0.0, F0
	CBZ R10, cr_dot_done
	VEOR V0.B16, V0.B16, V0.B16
	MOVD R10, R11
	LSR  $1, R11, R12
	CBZ  R12, cr_dot_tail
cr_dot_loop:
	VLD1.P 16(R8), [V1.D2]
	VLD1.P 16(R9), [V2.D2]
	VFMUL_D2(1, 2, 3)
	VFADD_D2(3, 0, 0)
	SUBS $1, R12, R12
	BNE  cr_dot_loop

	MOVD RSP, R12
	VST1.P [V0.D2], 16(R12)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0

cr_dot_tail:
	TST $1, R10
	BEQ cr_dot_done
	FMOVD (R8), F1
	FMOVD (R9), F2
	FMADDD F1, F0, F2, F0
cr_dot_done:
	MOVD R7, R3
	MUL R1, R3
	ADD R2, R3, R3
	LSL $3, R3, R3
	MOVD R0, R6
	ADD R3, R6, R6
	FMOVD (R6), F5
	FSUBD F0, F5, F5
	FMULD F4, F5, F5
	FMOVD F5, (R6)

	ADD $1, R7, R7
	B cr_row_loop

cr_next_col:
	ADD $1, R2, R2
	B cr_col_loop

cr_done:
	RET
