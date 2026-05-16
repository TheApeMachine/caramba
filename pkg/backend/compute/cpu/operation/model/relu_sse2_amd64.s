#include "textflag.h"

// reluSSE2(dst []float64) — in-place ReLU, 2-wide SSE2
TEXT ·reluSSE2(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP),     AX
	MOVQ dst_len+8(FP), BX

	XORPD X14, X14

loop2:
	CMPQ BX, $2
	JL   tail2

	MOVUPD (AX), X0
	MAXPD  X14, X0
	MOVUPD X0, (AX)
	ADDQ   $16, AX
	SUBQ   $2, BX
	JMP    loop2

tail2:
	CMPQ BX, $0
	JLE  done2

	MOVSD (AX), X0
	MAXSD X14, X0
	MOVSD X0, (AX)
	ADDQ  $8, AX
	DECQ  BX
	JMP   tail2

done2:
	RET
