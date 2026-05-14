#include "textflag.h"

DATA ·swigluConst27_amd64+0(SB)/8, $27.0
GLOBL ·swigluConst27_amd64(SB), RODATA, $8
DATA ·swigluConst9_amd64+0(SB)/8, $9.0
GLOBL ·swigluConst9_amd64(SB), RODATA, $8
DATA ·swigluHalf_amd64+0(SB)/8, $0.5
GLOBL ·swigluHalf_amd64(SB), RODATA, $8
DATA ·swigluOne_amd64+0(SB)/8, $1.0
GLOBL ·swigluOne_amd64(SB), RODATA, $8
DATA ·swigluNegOne_amd64+0(SB)/8, $-1.0
GLOBL ·swigluNegOne_amd64(SB), RODATA, $8

// SwiGLUAVX2(dst, src []float64)
// src.len = 2n; gates = src[0..n-1], values = src[n..2n-1]
// dst.len = n; dst[i] = gate[i] * sigmoid(gate[i]) * value[i]   (swish(gate) * value)
// ABI0: dst+0(FP)=ptr, dst_len+8(FP)=len(=n), dst_cap+16(FP)=cap,
//       src_base+24(FP)=ptr, src_len+32(FP)=len(=2n), src_cap+40(FP)=cap
TEXT ·SwiGLUAVX2(SB), NOSPLIT, $0-48
	MOVQ dst_len+8(FP), BX     // n = dst.len
	CMPQ BX, $0
	JLE  done

	MOVQ dst+0(FP), AX
	MOVQ src_base+24(FP), DI   // gates ptr

	VMOVSD ·swigluConst27_amd64(SB), X10
	VBROADCASTSD X10, Y10
	VMOVSD ·swigluConst9_amd64(SB), X11
	VBROADCASTSD X11, Y11
	VMOVSD ·swigluHalf_amd64(SB), X12
	VBROADCASTSD X12, Y12
	VMOVSD ·swigluOne_amd64(SB), X13
	VBROADCASTSD X13, Y13
	VMOVSD ·swigluNegOne_amd64(SB), X14
	VBROADCASTSD X14, Y14

	// values ptr = gates ptr + n*8
	MOVQ BX, R9
	SHLQ $3, R9
	ADDQ DI, R9

	MOVQ BX, SI               // use SI as element counter

loop:
	VMOVUPD (DI), Y0           // gates
	VMOVUPD (R9), Y1           // values
	VMULPD Y12, Y0, Y2         // gate/2
	VMULPD Y2, Y2, Y3
	VADDPD Y10, Y3, Y4
	VMULPD Y11, Y3, Y5
	VADDPD Y10, Y5, Y5
	VMULPD Y2, Y4, Y6
	VDIVPD Y5, Y6, Y6
	VMINPD Y13, Y6, Y6
	VMAXPD Y14, Y6, Y6
	VADDPD Y13, Y6, Y6
	VMULPD Y12, Y6, Y6         // sigmoid(gate)
	VMULPD Y0, Y6, Y6          // swish(gate) = gate * sigmoid(gate)
	VMULPD Y1, Y6, Y7          // swish(gate) * value
	VMOVUPD Y7, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	ADDQ $32, R9
	SUBQ $4, SI
	CMPQ SI, $4
	JGE  loop
done:
	VZEROUPPER
	RET
