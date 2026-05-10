#include "textflag.h"

// reduceSumNEON(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceSumNEON(SB), NOSPLIT, $0-32
	MOVD   a+0(FP), R0
	MOVD   a_len+8(FP), R1
	FMOVD  $0.0, F0
	LSR    $1, R1, R2
	CBZ    R2, done_rs
loop_rs:
	FMOVD.P 8(R0), F1
	FADDD   F1, F0, F0
	FMOVD.P 8(R0), F2
	FADDD   F2, F0, F0
	SUBS $1, R2, R2
	BNE  loop_rs
done_rs:
	FMOVD F0, ret+24(FP)
	RET

// reduceMaxNEON(a []float64) float64
TEXT ·reduceMaxNEON(SB), NOSPLIT, $0-32
	MOVD   a+0(FP), R0
	MOVD   a_len+8(FP), R1
	CBZ    R1, done_rm
	FMOVD.P 8(R0), F0
	SUBS $1, R1, R1
	CBZ  R1, done_rm
loop_rm:
	FMOVD.P 8(R0), F1
	FCMPD   F0, F1
	FCSELD  GT, F0, F1, F0
	SUBS $1, R1, R1
	BNE  loop_rm
done_rm:
	FMOVD F0, ret+24(FP)
	RET

// divScalarNEON(dst []float64, s float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP), s+24(FP)
TEXT ·divScalarNEON(SB), NOSPLIT, $0-32
	MOVD   dst+0(FP), R0
	MOVD   dst_len+8(FP), R1
	FMOVD  s+24(FP), F16
	LSR    $1, R1, R2
	CBZ    R2, done_ds
loop_ds:
	FMOVD.P 8(R0), F0
	FDIVD   F16, F0, F0
	FMOVD.P F0, 8(R0)
	FMOVD.P 8(R0), F1
	FDIVD   F16, F1, F1
	FMOVD.P F1, 8(R0)
	SUBS $1, R2, R2
	BNE  loop_ds
done_ds:
	RET

// addVecNEON(dst, a, b []float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP),
//       a+24(FP), a_len+32(FP), a_cap+40(FP),
//       b+48(FP), b_len+56(FP), b_cap+64(FP)
TEXT ·addVecNEON(SB), NOSPLIT, $0-72
	MOVD dst+0(FP), R0
	MOVD a_len+32(FP), R3
	MOVD a+24(FP), R1
	MOVD b+48(FP), R2
	LSR  $1, R3, R4
	CBZ  R4, done_av
loop_av:
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R2), F1
	FADDD   F1, F0, F0
	FMOVD.P F0, 8(R0)
	FMOVD.P 8(R1), F2
	FMOVD.P 8(R2), F3
	FADDD   F3, F2, F2
	FMOVD.P F2, 8(R0)
	SUBS $1, R4, R4
	BNE  loop_av
done_av:
	RET

// mulVecNEON(dst, a, b []float64)
TEXT ·mulVecNEON(SB), NOSPLIT, $0-72
	MOVD dst+0(FP), R0
	MOVD a_len+32(FP), R3
	MOVD a+24(FP), R1
	MOVD b+48(FP), R2
	LSR  $1, R3, R4
	CBZ  R4, done_mv
loop_mv:
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R2), F1
	FMULD   F1, F0, F0
	FMOVD.P F0, 8(R0)
	FMOVD.P 8(R1), F2
	FMOVD.P 8(R2), F3
	FMULD   F3, F2, F2
	FMOVD.P F2, 8(R0)
	SUBS $1, R4, R4
	BNE  loop_mv
done_mv:
	RET

// mulScalarNEON(dst []float64, s float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP), s+24(FP)
TEXT ·mulScalarNEON(SB), NOSPLIT, $0-32
	MOVD   dst+0(FP), R0
	MOVD   dst_len+8(FP), R1
	FMOVD  s+24(FP), F16
	LSR    $1, R1, R2
	CBZ    R2, done_ms
loop_ms:
	FMOVD.P 8(R0), F0
	FMULD   F16, F0, F0
	FMOVD.P F0, 8(R0)
	FMOVD.P 8(R0), F1
	FMULD   F16, F1, F1
	FMOVD.P F1, 8(R0)
	SUBS $1, R2, R2
	BNE  loop_ms
done_ms:
	RET

// reduceSumSqNEON(a []float64) float64
// Computes sum of squares
TEXT ·reduceSumSqNEON(SB), NOSPLIT, $0-32
	MOVD   a+0(FP), R0
	MOVD   a_len+8(FP), R1
	FMOVD  $0.0, F0
	LSR    $1, R1, R2
	CBZ    R2, done_ssq
loop_ssq:
	FMOVD.P 8(R0), F1
	FMADDD  F1, F1, F0, F0
	FMOVD.P 8(R0), F2
	FMADDD  F2, F2, F0, F0
	SUBS $1, R2, R2
	BNE  loop_ssq
done_ssq:
	FMOVD F0, ret+24(FP)
	RET
