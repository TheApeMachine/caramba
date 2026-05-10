#include "textflag.h"

// CopySSE2(dst, src []float64)
// Copies src into dst 2 float64s (16 bytes) per iteration using MOVUPD.
// ABI0: dst+0(FP)=ptr, src_base+24(FP)=ptr, src_len+32(FP)=len
TEXT ·CopySSE2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src_base+24(FP), DI
	MOVQ src_len+32(FP), BX
	CMPQ BX, $0
	JLE  done

loop:
	CMPQ BX, $2
	JL   scalar
	MOVUPD (DI), X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	JMP  loop

scalar:
	CMPQ BX, $0
	JLE  done
	MOVSD (DI), X0
	MOVSD X0, (AX)
	ADDQ $8, AX
	ADDQ $8, DI
	SUBQ $1, BX
	JMP  scalar

done:
	RET
