#include "textflag.h"

// RoPEHalfSSE2(dst, src, cosTable, sinTable []float64, numPairs int)
// Rotates split-half RoPE pairs: (i, i + numPairs).
TEXT ·RoPEHalfSSE2(SB), NOSPLIT, $0-104
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), SI
	MOVQ cosTable+48(FP), CX
	MOVQ sinTable+72(FP), DX
	MOVQ numPairs+96(FP), BX

	MOVQ BX, R10
	SHLQ $3, R10
	LEAQ (SI)(R10*1), R8
	LEAQ (AX)(R10*1), R9

vector_loop:
	CMPQ BX, $2
	JL scalar_tail

	MOVUPD (SI), X0
	MOVUPD (R8), X1
	MOVUPD (CX), X2
	MOVUPD (DX), X3

	MOVAPD X0, X4
	MULPD X2, X4
	MOVAPD X1, X5
	MULPD X3, X5
	SUBPD X5, X4

	MULPD X3, X0
	MULPD X2, X1
	ADDPD X1, X0

	MOVUPD X4, (AX)
	MOVUPD X0, (R9)

	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, AX
	ADDQ $16, R9
	ADDQ $16, CX
	ADDQ $16, DX
	SUBQ $2, BX
	JMP vector_loop

scalar_tail:
	CMPQ BX, $0
	JLE done

	MOVSD (SI), X0
	MOVSD (R8), X1
	MOVSD (CX), X2
	MOVSD (DX), X3

	MOVAPD X0, X4
	MULSD X2, X4
	MOVAPD X1, X5
	MULSD X3, X5
	SUBSD X5, X4
	MOVSD X4, (AX)

	MULSD X3, X0
	MULSD X2, X1
	ADDSD X1, X0
	MOVSD X0, (R9)

done:
	RET
