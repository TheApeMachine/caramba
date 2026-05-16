#include "textflag.h"

// upsampleNearest2DRowScale2SSE2(dst, src []float64)
TEXT ·upsampleNearest2DRowScale2SSE2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src_base+24(FP), DI
	MOVQ src_len+32(FP), BX
	CMPQ BX, $0
	JLE  done

loop:
	MOVSD (DI), X0
	UNPCKLPD X0, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $8, DI
	SUBQ $1, BX
	JMP  loop

done:
	RET
