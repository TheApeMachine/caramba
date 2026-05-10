#include "textflag.h"

// ApplyMaskSSE2(dst, scores, mask []float64)
// ABI0: dst+0(FP), ..., scores+24(FP), scores_len+32(FP), ..., mask+48(FP), ...
TEXT ·ApplyMaskSSE2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP),      AX
	MOVQ scores_len+32(FP), BX
	MOVQ scores+24(FP),  DI
	MOVQ mask+48(FP),    SI
	CMPQ BX, $2
	JL   am_sse2_scalar

am_sse2_loop:
	MOVUPD (DI), X0
	MOVUPD (SI), X1
	ADDPD  X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  am_sse2_loop

am_sse2_scalar:
	CMPQ BX, $0
	JLE  am_sse2_done
	MOVSD (DI), X0
	MOVSD (SI), X1
	ADDSD X1, X0
	MOVSD X0, (AX)

am_sse2_done:
	RET
