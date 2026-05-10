#include "textflag.h"

DATA ·sigConst27+0(SB)/8, $27.0
GLOBL ·sigConst27(SB), RODATA|NOPTR, $8
DATA ·sigConst9+0(SB)/8, $9.0
GLOBL ·sigConst9(SB), RODATA|NOPTR, $8
DATA ·sigHalf+0(SB)/8, $0.5
GLOBL ·sigHalf(SB), RODATA|NOPTR, $8
DATA ·sigOne+0(SB)/8, $1.0
GLOBL ·sigOne(SB), RODATA|NOPTR, $8

// SigmoidNEON(dst, src []float64)
// ABI0: dst+0(FP)=ptr, src_base+24(FP)=ptr, src_len+32(FP)=len

TEXT ·SigmoidNEON(SB),NOSPLIT,$0-48
	MOVD  dst+0(FP), R0
	MOVD  src_base+24(FP), R3
	MOVD  src_len+32(FP), R4
	FMOVD ·sigConst27(SB), F26
	FMOVD ·sigConst9(SB), F27
	FMOVD ·sigHalf(SB), F28
	FMOVD ·sigOne(SB), F29

	LSR  $1, R4, R5
	CBZ  R5, done

pairloop:
	FMOVD.P 8(R3), F0
	FMOVD.P 8(R3), F10
	// sigmoid A = 0.5*(1+tanh(x/2))
	FMULD F28, F0, F1      // x/2
	FMULD F1, F1, F2       // (x/2)^2
	FADDD F26, F2, F3      // 27+(x/2)^2
	FMULD F27, F2, F4      // 9*(x/2)^2
	FADDD F26, F4, F4      // 27+9*(x/2)^2
	FMULD F1, F3, F5       // (x/2)*(27+(x/2)^2)
	FDIVD F4, F5, F6       // tanh(x/2)
	FADDD F29, F6, F6      // 1+tanh
	FMULD F28, F6, F6      // 0.5*(1+tanh)
	// sigmoid B
	FMULD F28, F10, F11
	FMULD F11, F11, F12
	FADDD F26, F12, F13
	FMULD F27, F12, F14
	FADDD F26, F14, F14
	FMULD F11, F13, F15
	FDIVD F14, F15, F16
	FADDD F29, F16, F16
	FMULD F28, F16, F16
	FMOVD.P F6, 8(R0)
	FMOVD.P F16, 8(R0)
	SUBS $1, R5, R5
	BNE  pairloop

done:
	RET
