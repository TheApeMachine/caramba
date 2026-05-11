#include "textflag.h"

// Constants in tanh_avx2_amd64.s

// TanhSSE2(dst, src []float64)
// ABI0: dst+0(FP)=ptr, src_base+24(FP)=ptr, src_len+32(FP)=len
TEXT ·TanhSSE2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src_len+32(FP), SI
	MOVQ src_base+24(FP), DI
	CMPQ SI, $0
	JLE  done
	MOVSD  ·tanhConst27_amd64(SB), X10
	MOVDDUP X10, X10
	MOVSD  ·tanhConst9_amd64(SB), X11
	MOVDDUP X11, X11
	MOVSD  ·tanhOne_amd64(SB), X12
	MOVDDUP X12, X12
	MOVSD  ·tanhNegOne_amd64(SB), X13
	MOVDDUP X13, X13

loop:
	MOVUPD (DI), X0
	MOVAPD X0, X1
	MULPD  X0, X1
	MOVAPD X1, X2
	ADDPD  X10, X2
	MOVAPD X1, X3
	MULPD  X11, X3
	ADDPD  X10, X3
	MULPD  X0, X2
	MOVAPD X2, X4
	DIVPD  X3, X4
	MINPD  X12, X4
	MAXPD  X13, X4
	MOVUPD X4, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, SI
	CMPQ SI, $2
	JGE  loop
done:
	RET
