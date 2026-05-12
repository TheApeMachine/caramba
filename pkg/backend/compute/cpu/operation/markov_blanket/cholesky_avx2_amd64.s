#include "textflag.h"

DATA ·choleskyOne+0(SB)/8, $1.0
GLOBL ·choleskyOne(SB), RODATA, $8

// choleskyDecompAVX2(L []float64, n int) uint64
// In-place Cholesky factorisation; returns 1 on success, 0 on non-positive pivot.
//
// Size/alignment contract:
//   - n == 0 is a no-op success (the loop body never executes).
//   - n == 1 reduces to a scalar SQRTSD path (the SIMD loop has zero iterations
//     and the tail handles the single element).
//   - L has no alignment requirement — every load/store uses VMOVUPD/MOVUPD
//     (unaligned), so any byte-aligned pointer from `make([]float64, ...)`
//     works. SIMD vectorisation pays off starting at n >= 4 (one AVX2 lane
//     pass); below that the kernel still runs but reduces to scalar tail.
TEXT ·choleskyDecompAVX2(SB), NOSPLIT, $0-40
	MOVQ L+0(FP), AX
	MOVQ n+24(FP), DX
	XORQ R8, R8                               // col = 0

chol_col_loop:
	CMPQ R8, DX
	JGE chol_success

	// Compute Σ L[col*n + k]^2 for k = 0..col-1 (vectorized)
	MOVQ R8, BX
	IMULQ DX, BX                              // BX = col*n
	SHLQ $3, BX                               // BX *= 8
	MOVQ AX, SI
	ADDQ BX, SI                               // SI = &L[col*n + 0]
	MOVQ R8, R9                               // R9 = col (k-loop count)

	VXORPD Y0, Y0, Y0
	XORPD X1, X1
	MOVQ R9, CX
	CMPQ CX, $4
	JL chol_sumsq_tail
chol_sumsq_loop:
	VMOVUPD (SI), Y2
	VFMADD231PD Y2, Y2, Y0
	ADDQ $32, SI
	SUBQ $4, CX
	CMPQ CX, $4
	JGE chol_sumsq_loop
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
chol_sumsq_tail:
	CMPQ CX, $2
	JL chol_sumsq_scalar
	MOVUPD (SI), X2
	MULPD X2, X2
	HADDPD X2, X2
	ADDSD X2, X0
	ADDQ $16, SI
	SUBQ $2, CX
chol_sumsq_scalar:
	CMPQ CX, $0
	JLE chol_diag
	MOVSD (SI), X2
	MULSD X2, X2
	ADDSD X2, X0

chol_diag:
	// pivot = L[col*n + col] - X0
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
	DIVSD X10, X12                            // invDiag

	// For row in col+1..n: L[row*n+col] = (L[row*n+col] - Σ_k L[row*n+k]*L[col*n+k]) * invDiag
	MOVQ R8, R10
	INCQ R10                                  // row = col+1
chol_row_loop:
	CMPQ R10, DX
	JGE chol_next_col

	// dot = Σ L[row*n+k] * L[col*n+k] for k=0..col-1
	MOVQ R10, BX
	IMULQ DX, BX
	SHLQ $3, BX
	MOVQ AX, SI                               // SI = &L[row*n]
	ADDQ BX, SI

	MOVQ R8, BX
	IMULQ DX, BX
	SHLQ $3, BX
	MOVQ AX, DI                               // DI = &L[col*n]
	ADDQ BX, DI

	MOVQ R9, CX                               // k count = col
	VXORPD Y0, Y0, Y0
	CMPQ CX, $4
	JL chol_dot_tail
chol_dot_loop:
	VMOVUPD (SI), Y2
	VMOVUPD (DI), Y3
	VFMADD231PD Y2, Y3, Y0
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $4, CX
	CMPQ CX, $4
	JGE chol_dot_loop
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
chol_dot_tail:
	CMPQ CX, $2
	JL chol_dot_scalar
	MOVUPD (SI), X2
	MOVUPD (DI), X3
	MULPD X3, X2
	HADDPD X2, X2
	ADDSD X2, X0
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $2, CX
chol_dot_scalar:
	CMPQ CX, $0
	JLE chol_dot_done
	MOVSD (SI), X2
	MOVSD (DI), X3
	MULSD X3, X2
	ADDSD X2, X0
chol_dot_done:

	// L[row*n+col] = (current - dot) * invDiag
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
	VZEROUPPER
	RET

chol_fail:
	MOVQ $0, ret+32(FP)
	VZEROUPPER
	RET
