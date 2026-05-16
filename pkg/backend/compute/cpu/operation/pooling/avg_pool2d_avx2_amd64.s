#include "textflag.h"

// reduceSumAVX2(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceSumAVX2(SB), NOSPLIT, $0-32
	MOVQ   a+0(FP), AX
	MOVQ   a_len+8(FP), BX
	VXORPD Y0, Y0, Y0
	CMPQ   BX, $4
	JL     reduce_rs
loop_rs:
	VMOVUPD (AX), Y1
	VADDPD  Y1, Y0, Y0
	ADDQ $32, AX
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_rs

reduce_rs:
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0

tail_rs:
	CMPQ BX, $0
	JLE  done_rs
	VMOVSD (AX), X1
	VADDSD X1, X0, X0
	ADDQ $8, AX
	DECQ BX
	JMP  tail_rs

done_rs:
	MOVSD  X0, ret+24(FP)
	VZEROUPPER
	RET
