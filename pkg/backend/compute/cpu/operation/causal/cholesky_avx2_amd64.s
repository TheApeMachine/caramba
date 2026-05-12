#include "textflag.h"

// choleskyRegAVX2(L, A []float64, n int, eps float64)
// Computes in-place regularised lower Cholesky factor: starting from A (copied
// into L upon entry by the caller). If a pivot becomes non-positive, substitute
// `eps` (no panic).
TEXT ·choleskyRegAVX2(SB), NOSPLIT, $0-56
	MOVQ L+0(FP), AX
	MOVQ n+48(FP), DX
	MOVSD eps+56(FP), X20
	XORQ R8, R8

cr_col_loop:
	CMPQ R8, DX
	JGE cr_done

	// Σ L[col*n + k]^2
	MOVQ R8, BX
	IMULQ DX, BX
	SHLQ $3, BX
	MOVQ AX, SI
	ADDQ BX, SI
	MOVQ R8, R9
	VXORPD Y0, Y0, Y0
	MOVQ R9, CX
	CMPQ CX, $4
	JL cr_ss_tail
cr_ss_loop:
	VMOVUPD (SI), Y2
	VFMADD231PD Y2, Y2, Y0
	ADDQ $32, SI
	SUBQ $4, CX
	CMPQ CX, $4
	JGE cr_ss_loop
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
cr_ss_tail:
	CMPQ CX, $2
	JL cr_ss_scalar
	MOVUPD (SI), X2
	MULPD X2, X2
	HADDPD X2, X2
	ADDSD X2, X0
	ADDQ $16, SI
	SUBQ $2, CX
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
	MOVAPD X20, X10                           // clamp to eps
cr_pivot_ok:
	SQRTSD X10, X10
	MOVSD X10, (SI)
	MOVSD $1.0, X12
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
	VXORPD Y0, Y0, Y0
	CMPQ CX, $4
	JL cr_dot_tail
cr_dot_loop:
	VMOVUPD (SI), Y2
	VMOVUPD (DI), Y3
	VFMADD231PD Y2, Y3, Y0
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $4, CX
	CMPQ CX, $4
	JGE cr_dot_loop
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
cr_dot_tail:
	CMPQ CX, $2
	JL cr_dot_scalar
	MOVUPD (SI), X2
	MOVUPD (DI), X3
	MULPD X3, X2
	HADDPD X2, X2
	ADDSD X2, X0
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $2, CX
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
	VZEROUPPER
	RET
