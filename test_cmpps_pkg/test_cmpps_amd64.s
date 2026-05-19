#include "textflag.h"

TEXT ·testCMPPS(SB), NOSPLIT, $0-12
	MOVSS a+0(FP), X0
	MOVSS b+4(FP), X1
	CMPSS $6, X1, X0
	MOVD X0, AX
	MOVL AX, ret+8(FP)
	RET
