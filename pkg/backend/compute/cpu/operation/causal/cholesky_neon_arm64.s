#include "textflag.h"

// NOTE on the "NEON" suffix:
// The Go ARM64 assembler does not accept double-precision vector mnemonics
// (FMUL/FADD with .2D, FMLA.2D, FRINTN.2D, FCVTNS.2D, etc.) — only scalar
// FP64 instructions are available. This kernel therefore uses scalar FMOVD/
// FMADDD/FSQRTD throughout; vectorisation would require raw-WORD encodings
// of the NEON ops. The symbol name is kept for parity with the AVX2/SSE2
// dispatch on amd64.

// choleskyRegNEON(L []float64, n int, eps float64)
TEXT ·choleskyRegNEON(SB), NOSPLIT, $0-40
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
cr_ss_loop:
	FMOVD (R6), F1
	FMADDD F1, F0, F1, F0
	ADD $8, R6, R6
	SUBS $1, R5, R5
	BNE cr_ss_loop

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
cr_dot_loop:
	FMOVD (R8), F1
	FMOVD (R9), F2
	FMADDD F1, F0, F2, F0
	ADD $8, R8, R8
	ADD $8, R9, R9
	SUBS $1, R10, R10
	BNE cr_dot_loop
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
