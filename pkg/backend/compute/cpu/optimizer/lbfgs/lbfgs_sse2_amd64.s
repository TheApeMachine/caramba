#include "textflag.h"


TEXT ·lbfgsSubSSE2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ a+24(FP), R8
	MOVQ b+48(FP), R9
	MOVQ dst_len+8(FP), CX
	CMPQ CX, $2
	JL sub_sse2_tail
sub_sse2_loop:
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	SUBPD X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $2, CX
	CMPQ CX, $2
	JGE sub_sse2_loop
sub_sse2_tail:
	CMPQ CX, $0
	JLE sub_sse2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	SUBSD X1, X0
	MOVSD X0, (AX)
sub_sse2_done:
	RET


TEXT ·lbfgsDotSSE2(SB), NOSPLIT, $0-56
	MOVQ a+0(FP), AX
	MOVQ b+24(FP), R8
	MOVQ a_len+8(FP), CX
	XORPD X0, X0
	CMPQ CX, $2
	JL dot_sse2_tail
dot_sse2_loop:
	MOVUPD (AX), X1
	MOVUPD (R8), X2
	MULPD X2, X1
	ADDPD X1, X0
	ADDQ $16, AX
	ADDQ $16, R8
	SUBQ $2, CX
	CMPQ CX, $2
	JGE dot_sse2_loop
	MOVAPD X0, X1
	UNPCKHPD X0, X1
	ADDSD X1, X0
dot_sse2_tail:
	CMPQ CX, $0
	JLE dot_sse2_done
	MOVSD (AX), X1
	MOVSD (R8), X2
	MULSD X2, X1
	ADDSD X1, X0
dot_sse2_done:
	MOVSD X0, ret+48(FP)
	RET


TEXT ·lbfgsAddScaledSSE2(SB), NOSPLIT, $0-56
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), R8
	MOVQ dst_len+8(FP), CX
	MOVSD scale+48(FP), X8
	SHUFPD $0, X8, X8
	CMPQ CX, $2
	JL adsc_sse2_tail
adsc_sse2_loop:
	MOVUPD (AX), X0
	MOVUPD (R8), X1
	MULPD X8, X1
	ADDPD X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	SUBQ $2, CX
	CMPQ CX, $2
	JGE adsc_sse2_loop
adsc_sse2_tail:
	CMPQ CX, $0
	JLE adsc_sse2_done
	MOVSD (AX), X0
	MOVSD (R8), X1
	MULSD X8, X1
	ADDSD X1, X0
	MOVSD X0, (AX)
adsc_sse2_done:
	RET


TEXT ·lbfgsScaleSSE2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), AX
	MOVQ dst_len+8(FP), CX
	MOVSD scale+24(FP), X8
	SHUFPD $0, X8, X8
	CMPQ CX, $2
	JL sc_sse2_tail
sc_sse2_loop:
	MOVUPD (AX), X0
	MULPD X8, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	SUBQ $2, CX
	CMPQ CX, $2
	JGE sc_sse2_loop
sc_sse2_tail:
	CMPQ CX, $0
	JLE sc_sse2_done
	MOVSD (AX), X0
	MULSD X8, X0
	MOVSD X0, (AX)
sc_sse2_done:
	RET


TEXT ·lbfgsParamStepSSE2(SB), NOSPLIT, $0-80
	MOVQ out+0(FP), AX
	MOVQ params+24(FP), R8
	MOVQ dir+48(FP), R9
	MOVQ out_len+8(FP), CX
	MOVSD lr+72(FP), X8
	SHUFPD $0, X8, X8
	CMPQ CX, $2
	JL ps_sse2_tail
ps_sse2_loop:
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MULPD X8, X1
	SUBPD X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $2, CX
	CMPQ CX, $2
	JGE ps_sse2_loop
ps_sse2_tail:
	CMPQ CX, $0
	JLE ps_sse2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MULSD X8, X1
	SUBSD X1, X0
	MOVSD X0, (AX)
ps_sse2_done:
	RET
