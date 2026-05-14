#include "textflag.h"

// similarityKernelAVX2(a, b []float64) float64
TEXT ·similarityKernelAVX2(SB), NOSPLIT, $0-56
	MOVQ   a+0(FP), AX
	MOVQ   a_len+8(FP), BX
	MOVQ   b+24(FP), DI
	VXORPD Y0, Y0, Y0
	VXORPD Y5, Y5, Y5
	CMPQ   BX, $8
	JL     try4

loop8:
	VMOVUPD (AX), Y1
	VMOVUPD (DI), Y2
	VMULPD  Y2, Y1, Y1
	VADDPD  Y1, Y0, Y0
	VMOVUPD 32(AX), Y3
	VMOVUPD 32(DI), Y4
	VMULPD  Y4, Y3, Y3
	VADDPD  Y3, Y5, Y5
	ADDQ $64, AX
	ADDQ $64, DI
	SUBQ $8, BX
	CMPQ BX, $8
	JGE  loop8

try4:
	CMPQ BX, $4
	JL   reduce

loop4:
	VMOVUPD (AX), Y1
	VMOVUPD (DI), Y2
	VMULPD  Y2, Y1, Y1
	VADDPD  Y1, Y0, Y0
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop4

reduce:
	VADDPD Y5, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
	MOVSD  X0, ret+48(FP)
	VZEROUPPER
	RET
