#include "textflag.h"

// Constants live in swish_avx2_amd64.s.

// SwishSSE2(dst, src []float64)
// dst[i] = src[i] * sigmoid(src[i])
TEXT ·SwishSSE2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src_base+24(FP), DI
	MOVQ src_len+32(FP), SI
	CMPQ SI, $0
	JLE  done

	MOVSD  ·swishConst27_amd64(SB), X10
	MOVDDUP X10, X10
	MOVSD  ·swishConst9_amd64(SB), X11
	MOVDDUP X11, X11
	MOVSD  ·swishHalf_amd64(SB), X12
	MOVDDUP X12, X12
	MOVSD  ·swishOne_amd64(SB), X13
	MOVDDUP X13, X13
	MOVSD  ·swishNegOne_amd64(SB), X14
	MOVDDUP X14, X14

loop:
	MOVUPD (DI), X0
	MOVAPD X0, X1
	MULPD  X12, X1
	MOVAPD X1, X2
	MULPD  X1, X2
	MOVAPD X2, X3
	ADDPD  X10, X3
	MOVAPD X2, X4
	MULPD  X11, X4
	ADDPD  X10, X4
	MOVAPD X1, X5
	MULPD  X3, X5
	MOVAPD X5, X6
	DIVPD  X4, X6
	MINPD  X13, X6
	MAXPD  X14, X6
	ADDPD  X13, X6
	MULPD  X12, X6
	MULPD  X0, X6
	MOVUPD X6, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, SI
	CMPQ SI, $2
	JGE  loop

done:
	RET
