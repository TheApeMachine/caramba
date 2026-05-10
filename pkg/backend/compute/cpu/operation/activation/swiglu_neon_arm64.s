#include "textflag.h"

DATA ·swigluConst27+0(SB)/8, $27.0
GLOBL ·swigluConst27(SB), RODATA|NOPTR, $8
DATA ·swigluConst9+0(SB)/8, $9.0
GLOBL ·swigluConst9(SB), RODATA|NOPTR, $8
DATA ·swigluHalf+0(SB)/8, $0.5
GLOBL ·swigluHalf(SB), RODATA|NOPTR, $8
DATA ·swigluOne+0(SB)/8, $1.0
GLOBL ·swigluOne(SB), RODATA|NOPTR, $8

// SwiGLUNEON(dst, src []float64)
// src has 2n elements: gates[0..n-1] | values[n..2n-1]
// dst has n elements: sigmoid(gate[i]) * value[i]
// ABI0: dst+0(FP)=ptr, dst_len+8(FP)=len(=n), dst_cap+16(FP)=cap,
//       src_base+24(FP)=ptr, src_len+32(FP)=len(=2n), src_cap+40(FP)=cap

TEXT ·SwiGLUNEON(SB),NOSPLIT,$0-48
	MOVD  dst+0(FP), R0
	MOVD  dst_len+8(FP), R1    // n
	MOVD  src_base+24(FP), R3  // gates ptr

	FMOVD ·swigluConst27(SB), F26
	FMOVD ·swigluConst9(SB), F27
	FMOVD ·swigluHalf(SB), F28
	FMOVD ·swigluOne(SB), F29

	// values ptr = gates ptr + n*8
	LSL  $3, R1, R6
	ADD  R3, R6, R7

	CBZ  R1, done

loop:
	FMOVD.P 8(R3), F0    // gate
	FMOVD.P 8(R7), F20   // value
	// sigmoid(gate) = 0.5*(1+tanh(gate/2))
	FMULD F28, F0, F1    // gate/2
	FMULD F1, F1, F2     // (gate/2)^2
	FADDD F26, F2, F3    // 27+(gate/2)^2
	FMULD F27, F2, F4    // 9*(gate/2)^2
	FADDD F26, F4, F4    // 27+9*(gate/2)^2
	FMULD F1, F3, F5     // (gate/2)*(27+(gate/2)^2)
	FDIVD F4, F5, F6     // tanh(gate/2)
	FADDD F29, F6, F6    // 1+tanh
	FMULD F28, F6, F6    // sigmoid(gate)
	FMULD F20, F6, F7    // sigmoid*value
	FMOVD.P F7, 8(R0)
	SUBS $1, R1, R1
	BNE  loop
done:
	RET
