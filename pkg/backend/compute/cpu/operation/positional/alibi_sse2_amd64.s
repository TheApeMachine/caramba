//go:build amd64 && alibi_asm
// +build amd64,alibi_asm

#include "textflag.h"

// ALiBiRowSSE2(dst []float64, slope float64, q int, seqLenK int)
// Scalar loop using SSE2 scalar doubles.
// ABI0: dst+0, slope+24, q+32, seqLenK+40

TEXT ·ALiBiRowSSE2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP),      AX
	MOVQ slope+24(FP),   BX
	MOVQ q+32(FP),       CX
	MOVQ seqLenK+40(FP), DX

	MOVQ   BX, X0              // slope
	CVTSI2SD CX, X1             // float64(q)
	MULSD    X0, X1             // slope*q
	MOVSD    X1, X2             // save slope*q

	XORQ R8, R8                 // k = 0
loop:
	CMPQ DX, $0
	JLE  done

	MOVQ R8, R9
	CVTSI2SD R9, X3             // float64(k)
	MULSD X0, X3                // slope*k
	SUBSD X2, X3                // slope*k - slope*q = slope*(k-q)
	MOVSD X3, (AX)

	ADDQ $8, AX
	INCQ R8
	DECQ DX
	JNZ  loop
done:
	RET
