#include "textflag.h"

DATA ·sigConst27_amd64+0(SB)/8, $27.0
GLOBL ·sigConst27_amd64(SB), RODATA, $8
DATA ·sigConst9_amd64+0(SB)/8, $9.0
GLOBL ·sigConst9_amd64(SB), RODATA, $8
DATA ·sigHalf_amd64+0(SB)/8, $0.5
GLOBL ·sigHalf_amd64(SB), RODATA, $8
DATA ·sigOne_amd64+0(SB)/8, $1.0
GLOBL ·sigOne_amd64(SB), RODATA, $8

// SigmoidAVX2(dst, src []float64)
// ABI0: dst+0(FP)=ptr, src_base+24(FP)=ptr, src_len+32(FP)=len
// sigmoid(x) = 0.5*(1+tanh(x/2))
TEXT ·SigmoidAVX2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src_len+32(FP), SI
	MOVQ src_base+24(FP), DI
	CMPQ SI, $0
	JLE  done
	VMOVSD ·sigConst27_amd64(SB), X10
	VBROADCASTSD X10, Y10
	VMOVSD ·sigConst9_amd64(SB), X11
	VBROADCASTSD X11, Y11
	VMOVSD ·sigHalf_amd64(SB), X12
	VBROADCASTSD X12, Y12
	VMOVSD ·sigOne_amd64(SB), X13
	VBROADCASTSD X13, Y13

loop:
	VMOVUPD (DI), Y0
	VMULPD Y12, Y0, Y1           // x/2
	VMULPD Y1, Y1, Y2            // (x/2)^2
	VADDPD Y10, Y2, Y3           // 27+(x/2)^2
	VMULPD Y11, Y2, Y4           // 9*(x/2)^2
	VADDPD Y10, Y4, Y4           // 27+9*(x/2)^2
	VMULPD Y1, Y3, Y5            // (x/2)*(27+(x/2)^2)
	VDIVPD Y4, Y5, Y6            // tanh(x/2)
	VADDPD Y13, Y6, Y6           // 1+tanh
	VMULPD Y12, Y6, Y6           // 0.5*(1+tanh)
	VMOVUPD Y6, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, SI
	CMPQ SI, $4
	JGE  loop
done:
	VZEROUPPER
	RET
