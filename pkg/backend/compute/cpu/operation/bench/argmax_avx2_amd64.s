#include "textflag.h"

// argmaxAVX2(xs []float64) int
// Returns the index of the largest element.
TEXT ·argmaxAVX2(SB), NOSPLIT, $0-32
	MOVQ xs+0(FP), AX
	MOVQ xs_len+8(FP), CX
	XORQ BX, BX                              // best = 0
	CMPQ CX, $0
	JLE am_done
	MOVSD (AX), X0                            // best value
	XORQ DX, DX                               // i = 0
am_loop:
	INCQ DX
	CMPQ DX, CX
	JGE am_done
	MOVSD (AX)(DX*8), X1
	UCOMISD X0, X1
	JBE am_loop
	MOVAPD X1, X0
	MOVQ DX, BX
	JMP am_loop
am_done:
	MOVQ BX, ret+24(FP)
	RET
