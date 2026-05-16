#include "textflag.h"


TEXT ·hebbStepSSE2(SB), NOSPLIT, $0-80
	MOVQ out+0(FP), AX
	MOVQ params+24(FP), R8
	MOVQ grads+48(FP), R9
	MOVQ out_len+8(FP), CX
	MOVSD lr+72(FP), X8
	SHUFPD $0, X8, X8

	CMPQ CX, $2
	JL hebb_sse2_tail
hebb_sse2_loop:
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MULPD X8, X1
	ADDPD X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $2, CX
	CMPQ CX, $2
	JGE hebb_sse2_loop

hebb_sse2_tail:
	CMPQ CX, $0
	JLE hebb_sse2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MULSD X8, X1
	ADDSD X1, X0
	MOVSD X0, (AX)
hebb_sse2_done:
	RET


TEXT ·hebbStepNormSSE2(SB), NOSPLIT, $0-88
	MOVQ out+0(FP), AX
	MOVQ params+24(FP), R8
	MOVQ grads+48(FP), R9
	MOVQ out_len+8(FP), CX
	MOVSD lr+72(FP), X8
	SHUFPD $0, X8, X8
	XORPD X10, X10

	CMPQ CX, $2
	JL hebbn_sse2_tail
hebbn_sse2_loop:
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MULPD X8, X1
	ADDPD X1, X0
	MOVUPD X0, (AX)
	MOVAPD X0, X11
	MULPD X11, X11
	ADDPD X11, X10
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $2, CX
	CMPQ CX, $2
	JGE hebbn_sse2_loop

	MOVAPD X10, X11
	UNPCKHPD X10, X11
	ADDSD X11, X10

hebbn_sse2_tail:
	CMPQ CX, $0
	JLE hebbn_sse2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MULSD X8, X1
	ADDSD X1, X0
	MOVSD X0, (AX)
	MOVAPD X0, X11
	MULSD X11, X11
	ADDSD X11, X10

hebbn_sse2_done:
	SQRTSD X10, X10
	MOVSD X10, ret+80(FP)
	RET


TEXT ·hebbScaleSSE2(SB), NOSPLIT, $0-32
	MOVQ out+0(FP), AX
	MOVQ out_len+8(FP), CX
	MOVSD scale+24(FP), X8
	SHUFPD $0, X8, X8
	CMPQ CX, $2
	JL hsc_sse2_tail
hsc_sse2_loop:
	MOVUPD (AX), X0
	MULPD X8, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	SUBQ $2, CX
	CMPQ CX, $2
	JGE hsc_sse2_loop
hsc_sse2_tail:
	CMPQ CX, $0
	JLE hsc_sse2_done
	MOVSD (AX), X0
	MULSD X8, X0
	MOVSD X0, (AX)
hsc_sse2_done:
	RET


TEXT ·ojaStepSSE2(SB), NOSPLIT, $0-88
	MOVQ out+0(FP), AX
	MOVQ params+24(FP), R8
	MOVQ grads+48(FP), R9
	MOVQ out_len+8(FP), CX
	MOVSD lr+72(FP), X8
	SHUFPD $0, X8, X8
	MOVSD postSq+80(FP), X9
	SHUFPD $0, X9, X9
	MULPD X8, X9
	CMPQ CX, $2
	JL oja_sse2_tail
oja_sse2_loop:
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVAPD X0, X2
	MOVAPD X1, X10
	MULPD X8, X10
	ADDPD X10, X2
	MOVAPD X0, X10
	MULPD X9, X10
	SUBPD X10, X2
	MOVUPD X2, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $2, CX
	CMPQ CX, $2
	JGE oja_sse2_loop
oja_sse2_tail:
	CMPQ CX, $0
	JLE oja_sse2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVAPD X0, X2
	MOVAPD X1, X10
	MULSD X8, X10
	ADDSD X10, X2
	MOVAPD X0, X10
	MULSD X9, X10
	SUBSD X10, X2
	MOVSD X2, (AX)
oja_sse2_done:
	RET


TEXT ·reduceSumSqSSE2(SB), NOSPLIT, $0-32
	MOVQ a+0(FP), AX
	MOVQ a_len+8(FP), CX
	XORPD X0, X0
	CMPQ CX, $2
	JL rss_sse2_tail
rss_sse2_loop:
	MOVUPD (AX), X1
	MULPD X1, X1
	ADDPD X1, X0
	ADDQ $16, AX
	SUBQ $2, CX
	CMPQ CX, $2
	JGE rss_sse2_loop
	MOVAPD X0, X2
	UNPCKHPD X0, X2
	ADDSD X2, X0
rss_sse2_tail:
	CMPQ CX, $0
	JLE rss_sse2_done
	MOVSD (AX), X1
	MULSD X1, X1
	ADDSD X1, X0
rss_sse2_done:
	MOVSD X0, ret+24(FP)
	RET
