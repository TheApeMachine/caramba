#include "textflag.h"

// bindKernelSSE2(dst, a, b []float64)
TEXT ·bindKernelSSE2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ a_len+32(FP), BX
	MOVQ a+24(FP), DI
	MOVQ b+48(FP), SI
	CMPQ BX, $2
	JL   done

loop:
	MOVUPD (DI), X0
	MOVUPD (SI), X1
	MULPD  X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop

done:
	RET
