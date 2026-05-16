#include "textflag.h"

TEXT ·bundleAccumSSE2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), DI
	MOVQ dst_len+8(FP), CX
	CMPQ CX, $2
	JL bas_tail
bas_loop:
	MOVUPD (AX), X0
	MOVUPD (DI), X1
	ADDPD X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, CX
	CMPQ CX, $2
	JGE bas_loop
bas_tail:
	CMPQ CX, $0
	JLE bas_done
	MOVSD (AX), X0
	MOVSD (DI), X1
	ADDSD X1, X0
	MOVSD X0, (AX)
bas_done:
	RET

TEXT ·bundleNormalizeSSE2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), AX
	MOVQ dst_len+8(FP), CX
	MOVSD eps+24(FP), X10
	MOVQ AX, R8
	MOVQ CX, R9

	XORPD X0, X0
	CMPQ CX, $2
	JL bns_ss_tail
bns_ss_loop:
	MOVUPD (AX), X1
	MULPD X1, X1
	ADDPD X1, X0
	ADDQ $16, AX
	SUBQ $2, CX
	CMPQ CX, $2
	JGE bns_ss_loop
	MOVAPD X0, X1
	UNPCKHPD X0, X1
	ADDSD X1, X0
bns_ss_tail:
	CMPQ CX, $0
	JLE bns_have_sumsq
	MOVSD (AX), X1
	MULSD X1, X1
	ADDSD X1, X0
bns_have_sumsq:
	SQRTSD X0, X0
	UCOMISD X10, X0
	JBE bns_done
	MOVSD $1.0, X3
	DIVSD X0, X3
	MOVAPD X3, X11
	SHUFPD $0, X11, X11
	MOVAPD X3, X12

	MOVQ R8, AX
	MOVQ R9, CX
	CMPQ CX, $2
	JL bns_scale_tail
bns_scale_loop:
	MOVUPD (AX), X0
	MULPD X11, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	SUBQ $2, CX
	CMPQ CX, $2
	JGE bns_scale_loop
bns_scale_tail:
	CMPQ CX, $0
	JLE bns_done
	MOVSD (AX), X0
	MULSD X12, X0
	MOVSD X0, (AX)
bns_done:
	RET
