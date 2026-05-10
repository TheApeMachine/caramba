#include "textflag.h"

// Constants in sigmoid_avx2_amd64.s

// SigmoidSSE2(dst, src []float64)
// ABI0: dst+0(FP)=ptr, src_base+24(FP)=ptr, src_len+32(FP)=len
TEXT ·SigmoidSSE2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src_len+32(FP), SI
	MOVQ src_base+24(FP), DI
	CMPQ SI, $0
	JLE  done
	MOVSD  ·sigConst27_amd64(SB), X10
	MOVDDUP X10, X10
	MOVSD  ·sigConst9_amd64(SB), X11
	MOVDDUP X11, X11
	MOVSD  ·sigHalf_amd64(SB), X12
	MOVDDUP X12, X12
	MOVSD  ·sigOne_amd64(SB), X13
	MOVDDUP X13, X13

loop:
	MOVUPD (DI), X0
	MOVAPD X12, X1
	MULPD  X0, X1
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
	ADDPD  X13, X6
	MULPD  X12, X6
	MOVUPD X6, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, SI
	CMPQ SI, $2
	JGE  loop
done:
	RET
