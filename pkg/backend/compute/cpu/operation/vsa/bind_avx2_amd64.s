#include "textflag.h"

// bindKernelAVX2(dst, a, b []float64)
TEXT ·bindKernelAVX2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ a_len+32(FP), BX
	MOVQ a+24(FP), DI
	MOVQ b+48(FP), SI
	CMPQ BX, $4
	JL   done

loop:
	VMOVUPD (DI), Y0
	VMOVUPD (SI), Y1
	VMULPD  Y1, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop

done:
	VZEROUPPER
	RET
