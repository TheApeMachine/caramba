#include "textflag.h"

// choleskyRegSSE2(L []float64, n int, eps float64)
// SSE2 fallback for choleskyRegAVX2 — identical algorithm, 2-wide dot products
// instead of 4-wide, no FMA.
TEXT ·choleskyRegSSE2(SB), NOSPLIT, $0-40
	MOVQ L+0(FP), AX
	MOVQ n+24(FP), DX
	MOVSD eps+32(FP), X15
	XORQ R8, R8

cr_col_loop:
	CMPQ R8, DX
	JGE cr_done

	MOVQ R8, BX
	IMULQ DX, BX
	SHLQ $3, BX
	MOVQ AX, SI
	ADDQ BX, SI
	MOVQ R8, R9

	XORPD X0, X0
	MOVQ R9, CX
	CMPQ CX, $2
	JL cr_ss_scalar
cr_ss_loop:
	MOVUPD (SI), X2
	MULPD X2, X2
	ADDPD X2, X0
	ADDQ $16, SI
	SUBQ $2, CX
	CMPQ CX, $2
	JGE cr_ss_loop
	HADDPD X0, X0
cr_ss_scalar:
	CMPQ CX, $0
	JLE cr_diag
	MOVSD (SI), X2
	MULSD X2, X2
	ADDSD X2, X0

cr_diag:
	MOVQ R8, BX
	IMULQ DX, BX
	ADDQ R8, BX
	SHLQ $3, BX
	MOVQ AX, SI
	ADDQ BX, SI
	MOVSD (SI), X10
	SUBSD X0, X10
	XORPD X11, X11
	UCOMISD X11, X10
	JA cr_pivot_ok
	MOVAPD X15, X10
cr_pivot_ok:
	SQRTSD X10, X10
	MOVSD X10, (SI)
	MOVSD ·choleskyOne(SB), X12
	DIVSD X10, X12

	MOVQ R8, R10
	INCQ R10
cr_row_loop:
	CMPQ R10, DX
	JGE cr_next_col
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
	JL cr_dot_scalar
cr_dot_loop:
	MOVUPD (SI), X2
	MOVUPD (DI), X3
	MULPD X3, X2
	ADDPD X2, X0
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $2, CX
	CMPQ CX, $2
	JGE cr_dot_loop
	HADDPD X0, X0
cr_dot_scalar:
	CMPQ CX, $0
	JLE cr_dot_done
	MOVSD (SI), X2
	MOVSD (DI), X3
	MULSD X3, X2
	ADDSD X2, X0
cr_dot_done:
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
	JMP cr_row_loop

cr_next_col:
	INCQ R8
	JMP cr_col_loop

cr_done:
	RET
