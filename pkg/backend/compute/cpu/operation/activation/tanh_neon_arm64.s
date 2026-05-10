#include "textflag.h"

DATA ·tanhConst27+0(SB)/8, $27.0
GLOBL ·tanhConst27(SB), RODATA|NOPTR, $8
DATA ·tanhConst9+0(SB)/8, $9.0
GLOBL ·tanhConst9(SB), RODATA|NOPTR, $8

// TanhNEON(dst, src []float64)
// ABI0: dst+0(FP)=ptr, src_base+24(FP)=ptr, src_len+32(FP)=len

TEXT ·TanhNEON(SB),NOSPLIT,$0-48
	MOVD  dst+0(FP), R0
	MOVD  src_base+24(FP), R3
	MOVD  src_len+32(FP), R4
	FMOVD ·tanhConst27(SB), F28
	FMOVD ·tanhConst9(SB), F29

	LSR  $1, R4, R5
	CBZ  R5, done

pairloop:
	FMOVD.P 8(R3), F0
	FMOVD.P 8(R3), F10
	// tanh A: x*(27+x^2)/(27+9*x^2)
	FMULD F0, F0, F1
	FADDD F28, F1, F2
	FMULD F29, F1, F3
	FADDD F28, F3, F3
	FMULD F0, F2, F4
	FDIVD F3, F4, F5
	// tanh B
	FMULD F10, F10, F11
	FADDD F28, F11, F12
	FMULD F29, F11, F13
	FADDD F28, F13, F13
	FMULD F10, F12, F14
	FDIVD F13, F14, F15
	FMOVD.P F5, 8(R0)
	FMOVD.P F15, 8(R0)
	SUBS $1, R5, R5
	BNE  pairloop

done:
	RET
