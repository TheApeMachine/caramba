#include "textflag.h"

// matmulSSE2(dst, a, b []float64, M, K, N int)
// Row-major triple loop with a two-wide SSE2 column block.
TEXT ·matmulSSE2(SB), NOSPLIT, $0-96
	MOVQ dst+0(FP), R8
	MOVQ a+24(FP), R9
	MOVQ b+48(FP), R10
	MOVQ M+72(FP), R11
	MOVQ K+80(FP), R12
	MOVQ N+88(FP), R13

	TESTQ R11, R11
	JZ    mm2_done

	XORQ R14, R14

mm2_outer:
	CMPQ R14, R11
	JGE  mm2_done

	MOVQ  R14, AX
	IMULQ R12, AX
	MOVQ  R9, DI
	LEAQ  (DI)(AX*8), DI

	MOVQ  R14, AX
	IMULQ R13, AX
	MOVQ  R8, SI
	LEAQ  (SI)(AX*8), SI

	XORQ R15, R15

mm2_j2:
	MOVQ R13, CX
	SUBQ $2, CX
	CMPQ R15, CX
	JG   mm2_j1

	XORPD X0, X0

	XORQ AX, AX

mm2_k2:
	MOVSD    (DI)(AX*8), X15
	UNPCKLPD X15, X15

	MOVQ  AX, BX
	IMULQ R13, BX
	ADDQ  R15, BX
	MOVUPD (R10)(BX*8), X1

	MULPD X15, X1
	ADDPD X1, X0

	INCQ AX
	CMPQ AX, R12
	JL   mm2_k2

	MOVQ  SI, BX
	LEAQ  (BX)(R15*8), BX
	MOVUPD X0, (BX)

	ADDQ $2, R15
	JMP  mm2_j2

mm2_j1:
	CMPQ R15, R13
	JGE  mm2_next_i

	XORPD X0, X0

	XORQ AX, AX

mm2_k1:
	MOVSD (DI)(AX*8), X1

	MOVQ  AX, BX
	IMULQ R13, BX
	ADDQ  R15, BX
	MOVSD (R10)(BX*8), X2

	MULSD X2, X1
	ADDSD X1, X0

	INCQ AX
	CMPQ AX, R12
	JL   mm2_k1

	MOVQ SI, BX
	LEAQ (BX)(R15*8), BX
	MOVSD X0, (BX)

	INCQ R15
	JMP  mm2_j1

mm2_next_i:
	INCQ R14
	JMP  mm2_outer

mm2_done:
	RET
