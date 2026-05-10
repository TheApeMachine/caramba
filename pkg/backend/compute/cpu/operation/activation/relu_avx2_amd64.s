#include "textflag.h"

// ReLUAVX2(dst, src []float64)
// ABI0: dst+0(FP)=ptr, src_base+24(FP)=ptr, src_len+32(FP)=len
TEXT ·ReLUAVX2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src_len+32(FP), BX
	MOVQ src_base+24(FP), DI
	CMPQ BX, $0
	JLE  done
	VXORPS Y14, Y14, Y14

loop:
	VMOVUPD (DI), Y0
	VMAXPD  Y14, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop
done:
	VZEROUPPER
	RET
