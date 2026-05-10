#include "textflag.h"

// ReLUSSE2(dst, src []float64)
// ABI0: dst+0(FP)=ptr, src_base+24(FP)=ptr, src_len+32(FP)=len
TEXT ·ReLUSSE2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src_len+32(FP), BX
	MOVQ src_base+24(FP), DI
	CMPQ BX, $0
	JLE  done
	XORPS X14, X14

loop:
	MOVUPD (DI), X0
	MAXPD  X14, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop
done:
	RET
