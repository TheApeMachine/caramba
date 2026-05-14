#include "textflag.h"

// similarityKernelSSE2(a, b []float64) float64
TEXT ·similarityKernelSSE2(SB), NOSPLIT, $0-56
	MOVQ  a+0(FP), AX
	MOVQ  a_len+8(FP), BX
	MOVQ  b+24(FP), DI
	XORPD X0, X0
	CMPQ  BX, $2
	JL    done

loop:
	MOVUPD (AX), X1
	MOVUPD (DI), X2
	MULPD  X2, X1
	ADDPD  X1, X0
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop

done:
	HADDPD X0, X0
	MOVSD  X0, ret+48(FP)
	RET
