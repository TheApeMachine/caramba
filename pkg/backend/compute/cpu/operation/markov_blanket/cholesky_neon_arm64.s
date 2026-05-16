#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// choleskyDecompNEON(L []float64, n int) uint64
TEXT ·choleskyDecompNEON(SB), NOSPLIT, $16-40
	MOVD L+0(FP), R0
	MOVD n+24(FP), R1
	MOVD $0, R2                               // col = 0

chol_col_loop:
	CMP R1, R2
	BGE chol_success

	// rowPtr (col) = &L[col*n]
	MOVD R2, R3
	MUL R1, R3
	LSL $3, R3, R3
	MOVD R0, R4
	ADD R3, R4, R4                            // R4 = &L[col*n]
	MOVD R2, R5                               // k count = col

	FMOVD $0.0, F0
	MOVD R4, R6
	CBZ R5, chol_diag
	VEOR V0.B16, V0.B16, V0.B16
	MOVD R5, R11
	LSR  $1, R11, R12
	CBZ  R12, chol_sumsq_tail
chol_sumsq_loop:
	VLD1.P 16(R6), [V1.D2]
	VFMUL_D2(1, 1, 2)
	VFADD_D2(2, 0, 0)
	SUBS $1, R12, R12
	BNE  chol_sumsq_loop

	MOVD RSP, R12
	VST1.P [V0.D2], 16(R12)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0

chol_sumsq_tail:
	TST $1, R5
	BEQ chol_diag
	FMOVD (R6), F1
	FMADDD F1, F0, F1, F0

chol_diag:
	// SI = &L[col*n + col]
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
	// reflect (F2 - F3). BLE branches when F2 ≤ F3 (pivot ≤ 0 or NaN),
	// taking the chol_fail path. Otherwise we proceed to FSQRTD.
	FCMPD F3, F2
	BLE chol_fail
	FSQRTD F2, F2
	FMOVD F2, (R6)
	FMOVD $1.0, F4
	FDIVD F2, F4, F4                          // invDiag

	// rows col+1..n
	MOVD R2, R7
	ADD $1, R7, R7
chol_row_loop:
	CMP R1, R7
	BGE chol_next_col
	MOVD R7, R3
	MUL R1, R3
	LSL $3, R3, R3
	MOVD R0, R8
	ADD R3, R8, R8                            // &L[row*n]

	MOVD R2, R3
	MUL R1, R3
	LSL $3, R3, R3
	MOVD R0, R9
	ADD R3, R9, R9                            // &L[col*n]

	MOVD R2, R10                              // k count = col
	FMOVD $0.0, F0
	CBZ R10, chol_dot_done
	VEOR V0.B16, V0.B16, V0.B16
	MOVD R10, R11
	LSR  $1, R11, R12
	CBZ  R12, chol_dot_tail
chol_dot_loop:
	VLD1.P 16(R8), [V1.D2]
	VLD1.P 16(R9), [V2.D2]
	VFMUL_D2(1, 2, 3)
	VFADD_D2(3, 0, 0)
	SUBS $1, R12, R12
	BNE  chol_dot_loop

	MOVD RSP, R12
	VST1.P [V0.D2], 16(R12)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0

chol_dot_tail:
	TST $1, R10
	BEQ chol_dot_done
	FMOVD (R8), F1
	FMOVD (R9), F2
	FMADDD F1, F0, F2, F0
chol_dot_done:
	// L[row*n+col] = (current - dot) * invDiag
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
	B chol_row_loop

chol_next_col:
	ADD $1, R2, R2
	B chol_col_loop

chol_success:
	MOVD $1, R0
	MOVD R0, ret+32(FP)
	RET

chol_fail:
	MOVD $0, R0
	MOVD R0, ret+32(FP)
	RET
