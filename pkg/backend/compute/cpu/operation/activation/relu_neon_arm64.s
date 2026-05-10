#include "textflag.h"

DATA ·reluZero+0(SB)/8, $0.0
GLOBL ·reluZero(SB), RODATA|NOPTR, $8

// ReLUNEON(dst, src []float64)
// ABI0: dst+0(FP)=ptr, dst_len+8(FP)=len, dst_cap+16(FP)=cap,
//       src_base+24(FP)=ptr, src_len+32(FP)=len, src_cap+40(FP)=cap

TEXT ·ReLUNEON(SB),NOSPLIT,$0-48
	MOVD  dst+0(FP), R0
	MOVD  src_base+24(FP), R3
	MOVD  src_len+32(FP), R4
	FMOVD ·reluZero(SB), F31

	LSR  $1, R4, R5
	CBZ  R5, done

pairloop:
	FMOVD.P 8(R3), F0
	FMOVD.P 8(R3), F1
	FMAXD F31, F0, F0
	FMAXD F31, F1, F1
	FMOVD.P F0, 8(R0)
	FMOVD.P F1, 8(R0)
	SUBS $1, R5, R5
	BNE  pairloop

done:
	RET
