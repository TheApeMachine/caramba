#include "textflag.h"

// LeakyReLUNEON(dst, src []float64, alpha float64)
// ABI0: dst+0(FP)=ptr, dst_len+8(FP)=len, dst_cap+16(FP)=cap,
//       src_base+24(FP)=ptr, src_len+32(FP)=len, src_cap+40(FP)=cap,
//       alpha+48(FP)=float64
//
// out[i] = max(alpha*x, x)

TEXT ·LeakyReLUNEON(SB),NOSPLIT,$0-56
	MOVD  dst+0(FP), R0
	MOVD  src_base+24(FP), R3
	MOVD  src_len+32(FP), R4
	FMOVD alpha+48(FP), F30

	LSR  $1, R4, R5
	CBZ  R5, done

pairloop:
	FMOVD.P 8(R3), F1
	FMOVD.P 8(R3), F2
	FMULD F30, F1, F3
	FMULD F30, F2, F4
	FMAXD F3, F1, F5
	FMAXD F4, F2, F6
	FMOVD.P F5, 8(R0)
	FMOVD.P F6, 8(R0)
	SUBS $1, R5, R5
	BNE  pairloop

done:
	RET
