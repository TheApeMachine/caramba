#include "textflag.h"


TEXT ·sgdVanillaSSE2(SB), NOSPLIT, $0-88
	MOVQ out+0(FP), AX
	MOVQ params+24(FP), R8
	MOVQ grads+48(FP), R9
	MOVQ out_len+8(FP), CX
	MOVSD lr+72(FP), X8
	SHUFPD $0, X8, X8
	MOVSD wd+80(FP), X9
	SHUFPD $0, X9, X9

	CMPQ CX, $2
	JL   sgdv_sse2_tail
sgdv_sse2_loop:
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVAPD X0, X2
	MULPD X9, X2
	ADDPD X2, X1
	MULPD X8, X1
	SUBPD X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $2, CX
	CMPQ CX, $2
	JGE  sgdv_sse2_loop

sgdv_sse2_tail:
	CMPQ CX, $0
	JLE  sgdv_sse2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVAPD X0, X2
	MULSD X9, X2
	ADDSD X2, X1
	MULSD X8, X1
	SUBSD X1, X0
	MOVSD X0, (AX)
sgdv_sse2_done:
	RET


TEXT ·sgdMomentumSSE2(SB), NOSPLIT, $0-128
	MOVQ out+0(FP), AX
	MOVQ params+24(FP), R8
	MOVQ grads+48(FP), R9
	MOVQ velocity+72(FP), R10
	MOVQ out_len+8(FP), CX
	MOVSD lr+96(FP), X8
	SHUFPD $0, X8, X8
	MOVSD wd+104(FP), X9
	SHUFPD $0, X9, X9
	MOVSD momentum+112(FP), X10
	SHUFPD $0, X10, X10
	MOVQ nesterov+120(FP), DX

	CMPQ CX, $2
	JL   sgdm_sse2_tail
sgdm_sse2_loop:
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVUPD (R10), X2

	MOVAPD X1, X3
	MOVAPD X0, X4
	MULPD X9, X4
	ADDPD X4, X3
	MULPD X10, X2
	ADDPD X3, X2
	MOVUPD X2, (R10)

	CMPQ DX, $0
	JE sgdm_sse2_addV
	MOVAPD X2, X5
	MULPD X10, X5
	ADDPD X3, X5
	JMP sgdm_sse2_store
sgdm_sse2_addV:
	MOVAPD X2, X5
sgdm_sse2_store:
	MULPD X8, X5
	SUBPD X5, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $2, CX
	CMPQ CX, $2
	JGE  sgdm_sse2_loop

sgdm_sse2_tail:
	CMPQ CX, $0
	JLE sgdm_sse2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVSD (R10), X2

	MOVAPD X1, X3
	MOVAPD X0, X4
	MULSD X9, X4
	ADDSD X4, X3
	MULSD X10, X2
	ADDSD X3, X2
	MOVSD X2, (R10)

	CMPQ DX, $0
	JE sgdm_sse2_addVtail
	MOVAPD X2, X5
	MULSD X10, X5
	ADDSD X3, X5
	JMP sgdm_sse2_storeTail
sgdm_sse2_addVtail:
	MOVAPD X2, X5
sgdm_sse2_storeTail:
	MULSD X8, X5
	SUBSD X5, X0
	MOVSD X0, (AX)

sgdm_sse2_done:
	RET
