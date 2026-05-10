#include "textflag.h"

// LeakyReLUSSE2(dst, src []float64, alpha float64)
// ABI0: dst+0(FP)=ptr, src_base+24(FP)=ptr, src_len+32(FP)=len,
//       alpha+48(FP)=float64

TEXT ·LeakyReLUSSE2(SB), NOSPLIT, $0-56
	MOVQ dst+0(FP), AX
	MOVQ src_len+32(FP), SI
	MOVQ src_base+24(FP), DI
	CMPQ SI, $0
	JLE  done

	MOVSD alpha+48(FP), X0
	MOVDDUP X0, X4             // broadcast alpha

loop:
	MOVUPD  (DI), X1
	MOVAPD  X1, X2
	MULPD   X4, X2
	MAXPD   X2, X1
	MOVUPD  X1, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, SI
	CMPQ SI, $2
	JGE  loop
done:
	RET
