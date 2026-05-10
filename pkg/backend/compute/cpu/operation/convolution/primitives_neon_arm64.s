#include "textflag.h"

// dotProductNEON(a, b []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP),
//       b+24(FP), b_len+32(FP), b_cap+40(FP),
//       ret+48(FP)
TEXT ·dotProductNEON(SB), NOSPLIT, $0-56
	MOVD  a+0(FP), R0
	MOVD  a_len+8(FP), R2
	MOVD  b+24(FP), R1
	FMOVD $0.0, F0
	LSR   $1, R2, R3
	CBZ   R3, done_dp
loop_dp:
	FMOVD.P  8(R0), F1
	FMOVD.P  8(R1), F2
	FMADDD   F1, F2, F0, F0
	FMOVD.P  8(R0), F3
	FMOVD.P  8(R1), F4
	FMADDD   F3, F4, F0, F0
	SUBS $1, R3, R3
	BNE  loop_dp
done_dp:
	FMOVD F0, ret+48(FP)
	RET

// scaledAddNEON(dst, src []float64, scale float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP),
//       src+24(FP), src_len+32(FP), src_cap+40(FP),
//       scale+48(FP)
TEXT ·scaledAddNEON(SB), NOSPLIT, $0-56
	MOVD   dst+0(FP), R0
	MOVD   src_len+32(FP), R2
	MOVD   src+24(FP), R1
	FMOVD  scale+48(FP), F16
	LSR    $1, R2, R3
	CBZ    R3, done_sa
loop_sa:
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R1), F1
	FMOVD   (R0), F2
	FMULD   F16, F0, F0
	FADDD   F0, F2, F2
	FMOVD.P F2, 8(R0)
	FMOVD   (R0), F3
	FMULD   F16, F1, F1
	FADDD   F1, F3, F3
	FMOVD.P F3, 8(R0)
	SUBS $1, R3, R3
	BNE  loop_sa
done_sa:
	RET
