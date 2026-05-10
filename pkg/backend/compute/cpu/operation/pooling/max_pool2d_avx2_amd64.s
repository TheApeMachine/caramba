#include "textflag.h"

// reduceMaxAVX2(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceMaxAVX2(SB), NOSPLIT, $0-32
	MOVQ    a+0(FP), AX
	MOVQ    a_len+8(FP), BX
	CMPQ    BX, $4
	JL      scalar_rm
	VMOVUPD (AX), Y0
	ADDQ $32, AX
	SUBQ $4, BX
loop_rm:
	VMOVUPD (AX), Y1
	VMAXPD  Y1, Y0, Y0
	ADDQ $32, AX
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_rm
scalar_rm:
	VEXTRACTF128 $1, Y0, X1
	VMAXPD X1, X0, X0
	VUNPCKHPD X0, X0, X1
	VMAXSD X1, X0, X0
	MOVSD  X0, ret+24(FP)
	VZEROUPPER
	RET
