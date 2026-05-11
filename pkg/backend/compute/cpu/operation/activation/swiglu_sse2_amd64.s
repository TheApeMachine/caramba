#include "textflag.h"

// Constants in swiglu_avx2_amd64.s

// SwiGLUSSE2(dst, src []float64)
// ABI0: dst+0(FP)=ptr, dst_len+8(FP)=len(=n),
//       src_base+24(FP)=ptr, src_len+32(FP)=len(=2n)
TEXT ·SwiGLUSSE2(SB), NOSPLIT, $0-48
	MOVQ dst_len+8(FP), BX
	CMPQ BX, $0
	JLE  done

	MOVQ dst+0(FP), AX
	MOVQ src_base+24(FP), DI

	MOVSD  ·swigluConst27_amd64(SB), X10
	MOVDDUP X10, X10
	MOVSD  ·swigluConst9_amd64(SB), X11
	MOVDDUP X11, X11
	MOVSD  ·swigluHalf_amd64(SB), X12
	MOVDDUP X12, X12
	MOVSD  ·swigluOne_amd64(SB), X13
	MOVDDUP X13, X13
	MOVSD  ·swigluNegOne_amd64(SB), X14
	MOVDDUP X14, X14

	MOVQ BX, R9
	SHLQ $3, R9
	ADDQ DI, R9       // R9 = values ptr

	MOVQ BX, SI

loop:
	MOVUPD (DI), X0
	MOVUPD (R9), X1
	MOVAPD X12, X2
	MULPD  X0, X2
	MOVAPD X2, X3
	MULPD  X2, X3
	MOVAPD X3, X4
	ADDPD  X10, X4
	MOVAPD X3, X5
	MULPD  X11, X5
	ADDPD  X10, X5
	MOVAPD X2, X6
	MULPD  X4, X6
	MOVAPD X6, X7
	DIVPD  X5, X7
	// Clamp rational sigmoid approximation to (-1,+1) before +1 and *0.5 so lane values
	// stay in range when polynomial coefficients introduce slight overshoot.
	MINPD  X13, X7
	MAXPD  X14, X7
	ADDPD  X13, X7
	MULPD  X12, X7
	MULPD  X1, X7
	MOVUPD X7, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	ADDQ $16, R9
	SUBQ $2, SI
	CMPQ SI, $2
	JGE  loop
done:
	RET
