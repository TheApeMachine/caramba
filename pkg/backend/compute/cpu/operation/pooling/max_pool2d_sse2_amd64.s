#include "textflag.h"

// reduceMaxSSE2(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceMaxSSE2(SB), NOSPLIT, $0-32
	MOVQ   a+0(FP), AX
	MOVQ   a_len+8(FP), BX
	CMPQ   BX, $0
	JLE    zero_rm2
	CMPQ   BX, $2
	JL     scalar_rm2
	MOVUPD (AX), X0
	ADDQ $16, AX
	SUBQ $2, BX
loop_rm2:
	MOVUPD (AX), X1
	MAXPD  X1, X0
	ADDQ $16, AX
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_rm2
	MOVUPD X0, X1
	UNPCKHPD X0, X1
	MAXSD  X1, X0
	JMP tail_rm2

scalar_rm2:
	MOVSD (AX), X0
	ADDQ  $8, AX
	DECQ  BX

tail_rm2:
	CMPQ BX, $0
	JLE  done_rm2
	MOVSD (AX), X1
	MAXSD X1, X0
	ADDQ $8, AX
	DECQ BX
	JMP  tail_rm2

done_rm2:
	MOVSD  X0, ret+24(FP)
	RET

zero_rm2:
	XORPS X0, X0
	MOVSD X0, ret+24(FP)
	RET
