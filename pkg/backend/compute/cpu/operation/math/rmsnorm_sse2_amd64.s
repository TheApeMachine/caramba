#include "textflag.h"

// rmsNormRowSSE2(out, row, weight []float64, eps float64)
// Computes one RMSNorm row with two-wide SSE2 accumulation and normalization.
TEXT ·rmsNormRowSSE2(SB), NOSPLIT, $0-80
	MOVQ out+0(FP), AX
	MOVQ row+24(FP), R8
	MOVQ weight+48(FP), R9
	MOVQ out_len+8(FP), CX
	MOVQ CX, R11
	MOVQ CX, R12
	MOVQ R8, R13
	MOVQ R8, R14
	XORPD X0, X0

	CMPQ CX, $2
	JL   rmss_sum_tail

rmss_sum_loop:
	MOVUPD (R14), X1
	MULPD  X1, X1
	ADDPD  X1, X0
	ADDQ   $16, R14
	SUBQ   $2, CX
	CMPQ   CX, $2
	JGE    rmss_sum_loop
	HADDPD X0, X0

rmss_sum_tail:
	CMPQ CX, $0
	JLE  rmss_have_sum
	MOVSD (R14), X1
	MULSD X1, X1
	ADDSD X1, X0

rmss_have_sum:
	CVTSQ2SD R11, X1
	DIVSD X1, X0
	ADDSD eps+72(FP), X0
	SQRTSD X0, X0
	MOVSD  $1.0, X3
	DIVSD  X0, X3
	MOVAPD X3, X4
	SHUFPD $0, X4, X4

	MOVQ R13, R14
	MOVQ R12, CX
	CMPQ CX, $2
	JL   rmss_norm_tail

rmss_norm_loop:
	MOVUPD (R14), X1
	MULPD  X4, X1
	MOVUPD (R9), X2
	MULPD  X2, X1
	MOVUPD X1, (AX)
	ADDQ   $16, AX
	ADDQ   $16, R14
	ADDQ   $16, R9
	SUBQ   $2, CX
	CMPQ   CX, $2
	JGE    rmss_norm_loop

rmss_norm_tail:
	CMPQ CX, $0
	JLE  rmss_done
	MOVSD (R14), X1
	MULSD X3, X1
	MOVSD (R9), X2
	MULSD X2, X1
	MOVSD X1, (AX)

rmss_done:
	RET
