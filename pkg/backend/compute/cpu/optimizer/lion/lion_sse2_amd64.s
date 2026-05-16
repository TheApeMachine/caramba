#include "textflag.h"


TEXT ·lionStepSSE2(SB), NOSPLIT, $0-144
	MOVQ out+0(FP), AX
	MOVQ m+24(FP), R8
	MOVQ params+48(FP), R9
	MOVQ grads+72(FP), R10
	MOVQ out_len+8(FP), CX

	MOVSD lr+96(FP), X8
	SHUFPD $0, X8, X8
	MOVSD beta1+104(FP), X9
	SHUFPD $0, X9, X9
	MOVSD oneMinusBeta1+112(FP), X10
	SHUFPD $0, X10, X10
	MOVSD beta2+120(FP), X11
	SHUFPD $0, X11, X11
	MOVSD oneMinusBeta2+128(FP), X12
	SHUFPD $0, X12, X12
	MOVSD wd+136(FP), X13
	SHUFPD $0, X13, X13
	MOVSD ·lionOne(SB), X14
	SHUFPD $0, X14, X14
	MOVSD ·lionNegOne(SB), X15
	SHUFPD $0, X15, X15

	CMPQ CX, $2
	JL   lion_sse2_tail
lion_sse2_loop:
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVUPD (R10), X2

	MOVAPD X0, X3
	MULPD X9, X3
	MOVAPD X2, X4
	MULPD X10, X4
	ADDPD X4, X3                                // interp

	XORPD X6, X6
	MOVAPD X6, X4
	CMPPD X3, X4, $1                            // (0 < interp)
	MOVAPD X3, X5
	CMPPD X6, X5, $1                            // (interp < 0)
	ANDPD X14, X4
	ANDPD X15, X5
	ORPD X5, X4                                 // sign

	MOVAPD X1, X7
	MOVAPD X4, X5
	MULPD X8, X5
	SUBPD X5, X7
	MOVAPD X1, X4
	MULPD X13, X4
	MULPD X8, X4
	SUBPD X4, X7
	MOVUPD X7, (AX)

	MULPD X11, X0
	MOVAPD X2, X4
	MULPD X12, X4
	ADDPD X4, X0
	MOVUPD X0, (R8)

	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $2, CX
	CMPQ CX, $2
	JGE  lion_sse2_loop

lion_sse2_tail:
	CMPQ CX, $0
	JLE  lion_sse2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVSD (R10), X2

	MOVAPD X0, X3
	MULSD X9, X3
	MOVAPD X2, X4
	MULSD X10, X4
	ADDSD X4, X3

	XORPD X6, X6
	UCOMISD X6, X3
	JBE lion_sse2_not_pos
	MOVAPD X14, X4
	JMP lion_sse2_have_sign
lion_sse2_not_pos:
	UCOMISD X3, X6
	JBE lion_sse2_zero
	MOVAPD X15, X4
	JMP lion_sse2_have_sign
lion_sse2_zero:
	XORPD X4, X4
lion_sse2_have_sign:
	MOVAPD X1, X7
	MULSD X8, X4
	SUBSD X4, X7
	MOVAPD X1, X4
	MULSD X13, X4
	MULSD X8, X4
	SUBSD X4, X7
	MOVSD X7, (AX)

	MULSD X11, X0
	MOVAPD X2, X4
	MULSD X12, X4
	ADDSD X4, X0
	MOVSD X0, (R8)

lion_sse2_done:
	RET
