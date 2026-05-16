#include "textflag.h"

// matmulAVX2(dst, a, b []float64, M, K, N int)
// ABI0 FP layout:
//   dst+0(FP) ptr, +8 len, +16 cap
//   a+24(FP)  ptr, +32 len, +40 cap
//   b+48(FP)  ptr, +56 len, +64 cap
//   M+72(FP), K+80(FP), N+88(FP)
//
// Row-major triple loop. Inner j-loop vectorised 4-wide using AVX2 FMA.
// For each row i of A, for each column block j of B (width 4), accumulate.
// Tail handles N not a multiple of 4.

TEXT ·matmulAVX2(SB), NOSPLIT, $0-96
	MOVQ dst+0(FP), R8    // dst ptr
	MOVQ a+24(FP),  R9    // A ptr
	MOVQ b+48(FP),  R10   // B ptr
	MOVQ M+72(FP),  R11   // M
	MOVQ K+80(FP),  R12   // K
	MOVQ N+88(FP),  R13   // N

	TESTQ R11, R11
	JZ    mm_done

	XORQ R14, R14          // i = 0
mm_outer:
	CMPQ R14, R11
	JGE  mm_done

	// Compute base offset for row i of A: aRowOff = i * K (in elements)
	// and row i of C: cRowOff = i * N
	// We'll compute pointers directly.
	MOVQ R14, AX
	IMULQ R12, AX          // AX = i*K
	// A row i ptr = R9 + AX*8
	MOVQ R9, DI
	LEAQ (DI)(AX*8), DI    // DI = &A[i*K]

	MOVQ R14, AX
	IMULQ R13, AX          // AX = i*N
	MOVQ R8, SI
	LEAQ (SI)(AX*8), SI    // SI = &C[i*N]

	XORQ R15, R15          // j = 0

	// j-loop, 4-wide AVX2
mm_j4:
	MOVQ R13, CX
	SUBQ $4, CX
	CMPQ R15, CX
	JG   mm_j1

	VXORPD Y0, Y0, Y0      // acc[0..3] = 0

	XORQ AX, AX            // k = 0
mm_k4:
	// Broadcast A[i,k]
	VMOVSD (DI)(AX*8), X15
	VBROADCASTSD X15, Y15

	// Load B[k, j..j+3]
	MOVQ AX, BX
	IMULQ R13, BX
	ADDQ R15, BX
	VMOVUPD (R10)(BX*8), Y1

	VFMADD231PD Y15, Y1, Y0

	INCQ AX
	CMPQ AX, R12
	JL   mm_k4

	// Store C[i, j..j+3]
	MOVQ SI, BX
	LEAQ (BX)(R15*8), BX
	VMOVUPD Y0, (BX)

	ADDQ $4, R15
	JMP  mm_j4

	// j-loop, scalar tail
mm_j1:
	CMPQ R15, R13
	JGE  mm_next_i

	VXORPD X0, X0, X0      // acc = 0

	XORQ AX, AX            // k = 0
mm_k1:
	VMOVSD (DI)(AX*8), X1  // A[i,k]

	MOVQ AX, BX
	IMULQ R13, BX
	ADDQ R15, BX
	VMOVSD (R10)(BX*8), X2 // B[k,j]

	VFMADD231SD X2, X1, X0

	INCQ AX
	CMPQ AX, R12
	JL   mm_k1

	// Store C[i, j]
	MOVQ SI, BX
	LEAQ (BX)(R15*8), BX
	VMOVSD X0, (BX)

	INCQ R15
	JMP  mm_j1

mm_next_i:
	INCQ R14
	JMP  mm_outer

mm_done:
	VZEROUPPER
	RET
