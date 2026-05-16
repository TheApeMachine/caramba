#include "textflag.h"

// choleskyDecompSSE2(L []float64, n int) uint64
// SSE2 fallback for choleskyDecompAVX2 — identical algorithm, 2-wide.
TEXT ·choleskyDecompSSE2(SB), NOSPLIT, $0-40
	MOVQ L+0(FP), AX
	MOVQ n+24(FP), DX
	XORQ R8, R8

chol_col_loop:
	CMPQ R8, DX
	JGE chol_success

	MOVQ R8, BX
	IMULQ DX, BX
	SHLQ $3, BX
	MOVQ AX, SI
	ADDQ BX, SI
	MOVQ R8, R9

	XORPD X0, X0
	MOVQ R9, CX
	CMPQ CX, $2
	JL chol_sumsq_scalar
chol_sumsq_loop:
	MOVUPD (SI), X2
	MULPD X2, X2
	ADDPD X2, X0
	ADDQ $16, SI
	SUBQ $2, CX
	CMPQ CX, $2
	JGE chol_sumsq_loop
	MOVAPD X0, X1
	UNPCKHPD X0, X1
	ADDSD X1, X0
chol_sumsq_scalar:
	CMPQ CX, $0
	JLE chol_diag
	MOVSD (SI), X2
	MULSD X2, X2
	ADDSD X2, X0

chol_diag:
	MOVQ AX, SI
	MOVQ R8, BX
	IMULQ DX, BX
	ADDQ R8, BX
	SHLQ $3, BX
	ADDQ BX, SI
	MOVSD (SI), X10
	SUBSD X0, X10
	XORPD X11, X11
	UCOMISD X11, X10
	JBE chol_fail
	SQRTSD X10, X10
	MOVSD X10, (SI)
	MOVSD ·choleskyOne(SB), X12
	DIVSD X10, X12

	MOVQ R8, R10
	INCQ R10
chol_row_loop:
	CMPQ R10, DX
	JGE chol_next_col

	MOVQ R10, BX
	IMULQ DX, BX
	SHLQ $3, BX
	MOVQ AX, SI
	ADDQ BX, SI

	MOVQ R8, BX
	IMULQ DX, BX
	SHLQ $3, BX
	MOVQ AX, DI
	ADDQ BX, DI

	MOVQ R9, CX
	XORPD X0, X0
	CMPQ CX, $2
	JL chol_dot_scalar
chol_dot_loop:
	MOVUPD (SI), X2
	MOVUPD (DI), X3
	MULPD X3, X2
	ADDPD X2, X0
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $2, CX
	CMPQ CX, $2
	JGE chol_dot_loop
	MOVAPD X0, X2
	UNPCKHPD X0, X2
	ADDSD X2, X0
chol_dot_scalar:
	CMPQ CX, $0
	JLE chol_dot_done
	MOVSD (SI), X2
	MOVSD (DI), X3
	MULSD X3, X2
	ADDSD X2, X0
chol_dot_done:

	MOVQ R10, BX
	IMULQ DX, BX
	ADDQ R8, BX
	SHLQ $3, BX
	MOVQ AX, SI
	ADDQ BX, SI
	MOVSD (SI), X4
	SUBSD X0, X4
	MULSD X12, X4
	MOVSD X4, (SI)

	INCQ R10
	JMP chol_row_loop

chol_next_col:
	INCQ R8
	JMP chol_col_loop

chol_success:
	MOVQ $1, ret+32(FP)
	RET

chol_fail:
	MOVQ $0, ret+32(FP)
	RET
