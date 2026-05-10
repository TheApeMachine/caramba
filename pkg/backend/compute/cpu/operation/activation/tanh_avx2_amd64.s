#include "textflag.h"

DATA ·tanhConst27_amd64+0(SB)/8, $27.0
GLOBL ·tanhConst27_amd64(SB), RODATA, $8
DATA ·tanhConst9_amd64+0(SB)/8, $9.0
GLOBL ·tanhConst9_amd64(SB), RODATA, $8

// TanhAVX2(dst, src []float64)
// ABI0: dst+0(FP)=ptr, src_base+24(FP)=ptr, src_len+32(FP)=len
TEXT ·TanhAVX2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src_len+32(FP), SI
	MOVQ src_base+24(FP), DI
	CMPQ SI, $0
	JLE  done
	VMOVSD ·tanhConst27_amd64(SB), X10
	VBROADCASTSD X10, Y10
	VMOVSD ·tanhConst9_amd64(SB), X11
	VBROADCASTSD X11, Y11

loop:
	VMOVUPD (DI), Y0
	VMULPD Y0, Y0, Y1
	VADDPD Y10, Y1, Y2
	VMULPD Y11, Y1, Y3
	VADDPD Y10, Y3, Y3
	VMULPD Y0, Y2, Y4
	VDIVPD Y3, Y4, Y5
	VMOVUPD Y5, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, SI
	CMPQ SI, $4
	JGE  loop
done:
	VZEROUPPER
	RET
