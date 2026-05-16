#include "textflag.h"

// reduceSumSSE2(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceSumSSE2(SB), NOSPLIT, $0-32
	MOVQ   a+0(FP), AX
	MOVQ   a_len+8(FP), BX
	XORPS  X0, X0
	CMPQ   BX, $2
	JL     reduce_rs2
loop_rs2:
	MOVUPD (AX), X1
	ADDPD  X1, X0
	ADDQ $16, AX
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_rs2

reduce_rs2:
	MOVAPD X0, X1
	UNPCKHPD X0, X1
	ADDSD X1, X0

tail_rs2:
	CMPQ BX, $0
	JLE  done_rs2
	MOVSD (AX), X1
	ADDSD X1, X0
	ADDQ $8, AX
	DECQ BX
	JMP  tail_rs2

done_rs2:
	MOVSD  X0, ret+24(FP)
	RET
