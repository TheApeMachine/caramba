#include "textflag.h"

// layerNormRowSSE2(out, row, weight, bias []float64, eps float64)
// Single LayerNorm row using two-lane SSE2 accumulation and normalization.
TEXT ·layerNormRowSSE2(SB), NOSPLIT, $0-104
	MOVQ out+0(FP), AX
	MOVQ row+24(FP), R8
	MOVQ weight+48(FP), R9
	MOVQ bias+72(FP), R10
	MOVQ out_len+8(FP), CX
	MOVQ CX, R11
	MOVQ CX, R12
	MOVQ R8, R13
	MOVQ R8, R14
	XORPD X0, X0

	CMPQ CX, $2
	JL   lns_sum_tail

lns_sum_loop:
	MOVUPD (R14), X1
	ADDPD  X1, X0
	ADDQ   $16, R14
	SUBQ   $2, CX
	CMPQ   CX, $2
	JGE    lns_sum_loop
	MOVAPD X0, X1
	UNPCKHPD X0, X1
	ADDSD X1, X0

lns_sum_tail:
	CMPQ CX, $0
	JLE  lns_have_sum
	MOVSD (R14), X1
	ADDSD X1, X0

lns_have_sum:
	CVTSQ2SD R11, X1
	DIVSD X1, X0
	MOVAPD X0, X3
	SHUFPD $0, X3, X3

	MOVQ R13, R14
	MOVQ R12, CX
	XORPD X4, X4

	CMPQ CX, $2
	JL   lns_var_tail

lns_var_loop:
	MOVUPD (R14), X1
	SUBPD  X3, X1
	MULPD  X1, X1
	ADDPD  X1, X4
	ADDQ   $16, R14
	SUBQ   $2, CX
	CMPQ   CX, $2
	JGE    lns_var_loop
	MOVAPD X4, X5
	UNPCKHPD X4, X5
	ADDSD X5, X4

lns_var_tail:
	CMPQ CX, $0
	JLE  lns_have_var
	MOVSD  (R14), X1
	MOVAPD X3, X5
	SUBSD  X5, X1
	MULSD  X1, X1
	ADDSD  X1, X4

lns_have_var:
	CVTSQ2SD R11, X5
	DIVSD X5, X4
	ADDSD eps+96(FP), X4
	SQRTSD X4, X4
	MOVSD  $1.0, X6
	DIVSD  X4, X6
	MOVAPD X6, X7
	SHUFPD $0, X7, X7

	MOVQ R13, R14
	MOVQ R12, CX
	CMPQ CX, $2
	JL   lns_norm_tail

lns_norm_loop:
	MOVUPD (R14), X1
	SUBPD  X3, X1
	MULPD  X7, X1
	MOVUPD (R9), X2
	MULPD  X2, X1
	MOVUPD (R10), X2
	ADDPD  X2, X1
	MOVUPD X1, (AX)
	ADDQ   $16, AX
	ADDQ   $16, R14
	ADDQ   $16, R9
	ADDQ   $16, R10
	SUBQ   $2, CX
	CMPQ   CX, $2
	JGE    lns_norm_loop

lns_norm_tail:
	CMPQ CX, $0
	JLE  lns_done
	MOVSD  (R14), X1
	MOVAPD X3, X5
	SUBSD  X5, X1
	MOVAPD X7, X8
	MULSD  X8, X1
	MOVSD  (R9), X2
	MULSD  X2, X1
	MOVSD  (R10), X2
	ADDSD  X2, X1
	MOVSD  X1, (AX)

lns_done:
	RET
