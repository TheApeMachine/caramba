#include "textflag.h"

DATA ·primZero+0(SB)/8, $0.0
GLOBL ·primZero(SB), RODATA|NOPTR, $8

// dotProductNEON(a, b []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP),
//       b+24(FP), b_len+32(FP), b_cap+40(FP),
//       ret+48(FP)
TEXT ·dotProductNEON(SB), NOSPLIT, $0-56
	MOVD  a+0(FP),   R0
	MOVD  a_len+8(FP), R2
	MOVD  b+24(FP),  R1
	FMOVD ·primZero(SB), F0
	CBZ   R2, done_dp
loop_dp:
	FMOVD.P 8(R0), F1
	FMOVD.P 8(R1), F2
	FMADDD  F1, F0, F2, F0
	SUBS $1, R2, R2
	BNE  loop_dp
done_dp:
	FMOVD F0, ret+48(FP)
	RET

// scaledAddNEON(dst, src []float64, scale float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP),
//       src+24(FP), src_len+32(FP), src_cap+40(FP),
//       scale+48(FP)
TEXT ·scaledAddNEON(SB), NOSPLIT, $0-56
	MOVD   dst+0(FP),    R0
	MOVD   dst_len+8(FP), R2
	MOVD   src+24(FP),   R1
	FMOVD  scale+48(FP), F16
	CBZ    R2, done_sa
loop_sa:
	FMOVD.P 8(R1), F0
	FMOVD   (R0),  F2
	FMULD   F16, F0, F0
	FADDD   F0, F2, F2
	FMOVD.P F2, 8(R0)
	SUBS $1, R2, R2
	BNE  loop_sa
done_sa:
	RET

// reduceMaxNEON(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceMaxNEON(SB), NOSPLIT, $0-32
	MOVD   a+0(FP),    R0
	MOVD   a_len+8(FP), R1
	CBZ    R1, done_rm
	FMOVD.P 8(R0), F0
	SUBS $1, R1, R1
	CBZ  R1, done_rm
loop_rm:
	FMOVD.P 8(R0), F1
	FCMPD   F0, F1
	FCSELD  GT, F1, F0, F0
	SUBS $1, R1, R1
	BNE  loop_rm
done_rm:
	FMOVD F0, ret+24(FP)
	RET

// reduceSumNEON(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceSumNEON(SB), NOSPLIT, $0-32
	MOVD   a+0(FP),    R0
	MOVD   a_len+8(FP), R1
	FMOVD  ·primZero(SB), F0
	CBZ    R1, done_rs
loop_rs:
	FMOVD.P 8(R0), F1
	FADDD   F1, F0, F0
	SUBS $1, R1, R1
	BNE  loop_rs
done_rs:
	FMOVD F0, ret+24(FP)
	RET

// divScalarNEON(dst []float64, s float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP), s+24(FP)
TEXT ·divScalarNEON(SB), NOSPLIT, $0-32
	MOVD   dst+0(FP),    R0
	MOVD   dst_len+8(FP), R1
	FMOVD  s+24(FP), F16
	CBZ    R1, done_ds
loop_ds:
	FMOVD   (R0), F0
	FDIVD   F16, F0, F0
	FMOVD.P F0, 8(R0)
	SUBS $1, R1, R1
	BNE  loop_ds
done_ds:
	RET
