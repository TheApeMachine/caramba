#include "textflag.h"

// mergeHeadsCopyAVX2(dst, src []float64)
TEXT ·mergeHeadsCopyAVX2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src_base+24(FP), DI
	MOVQ src_len+32(FP), BX
	CMPQ BX, $0
	JLE  done

loop:
	CMPQ BX, $4
	JL   scalar
	VMOVUPD (DI), Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, BX
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
	VZEROUPPER
	RET
