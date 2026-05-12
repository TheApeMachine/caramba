#include "textflag.h"

// RoPESSE2(dst, src, cosTable, sinTable []float64, numPairs int)
// Processes 2 pairs (4 float64s) per iteration using SSE2.

TEXT ·RoPESSE2(SB), NOSPLIT, $0-104
	MOVQ dst+0(FP),      AX
	MOVQ src+24(FP),     SI
	MOVQ cosTable+48(FP),     CX
	MOVQ sinTable+72(FP),     DX
	MOVQ numPairs+96(FP), BX

loop:
	CMPQ BX, $0
	JLE  done

	MOVSD (SI),   X0   // x[2i]
	MOVSD 8(SI),  X1   // x[2i+1]
	MOVSD (CX),   X2   // cos[i]
	MOVSD (DX),   X3   // sin[i]

	// d[2i] = x0*cos - x1*sin
	MOVAPD X0, X4
	MULSD  X2, X4
	MOVAPD X1, X5
	MULSD  X3, X5
	SUBSD  X5, X4
	MOVSD  X4, (AX)

	// d[2i+1] = x0*sin + x1*cos
	MULSD  X3, X0
	MULSD  X2, X1
	ADDSD  X1, X0
	MOVSD  X0, 8(AX)

	ADDQ $16, SI
	ADDQ $16, AX
	ADDQ $8,  CX
	ADDQ $8,  DX
	DECQ BX
	JNZ  loop

done:
	RET
