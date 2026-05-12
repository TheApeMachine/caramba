#include "textflag.h"

// ARM64 *NEON entrypoints: 2-wide scalar-FP unrolling for throughput (FMOVD.P / FMULD / FADDD).
// Names retain the NEON suffix as the platform SIMD path; they are not LD1/ST1 vector-lane kernels.

// bindNEON(dst, a, b []float64)
// ABI0: dst+0(FP)..16, a+24(FP)..40, b+48(FP)..64
TEXT ·bindNEON(SB), NOSPLIT, $0-72
	MOVD a+24(FP), R1
	MOVD a_len+32(FP), R3
	MOVD b+48(FP), R2
	MOVD dst+0(FP), R0
	LSR  $1, R3, R4
	CBZ  R4, done_bn
loop_bn:
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R2), F2
	FMULD   F2, F0, F0
	FMOVD.P F0, 8(R0)
	FMOVD.P 8(R1), F1
	FMOVD.P 8(R2), F3
	FMULD   F3, F1, F1
	FMOVD.P F1, 8(R0)
	SUBS $1, R4, R4
	BNE  loop_bn
done_bn:
	RET

// dotReduceNEON(a, b []float64) float64
// ABI0: a+0(FP)..16, b+24(FP)..40, ret+48(FP)
TEXT ·dotReduceNEON(SB), NOSPLIT, $0-56
	MOVD  a+0(FP), R0
	MOVD  a_len+8(FP), R1
	MOVD  b+24(FP), R2
	FMOVD $0.0, F0
	FMOVD $0.0, F5
	LSR   $2, R1, R3
	CBZ   R3, try_pair_dn
loop_quad_dn:
	FMOVD.P 8(R0), F1
	FMOVD.P 8(R2), F3
	FMULD   F3, F1, F1
	FADDD   F1, F0, F0
	FMOVD.P 8(R0), F2
	FMOVD.P 8(R2), F4
	FMULD   F4, F2, F2
	FADDD   F2, F5, F5
	FMOVD.P 8(R0), F6
	FMOVD.P 8(R2), F7
	FMULD   F7, F6, F6
	FADDD   F6, F0, F0
	FMOVD.P 8(R0), F8
	FMOVD.P 8(R2), F9
	FMULD   F9, F8, F8
	FADDD   F8, F5, F5
	SUBS $1, R3, R3
	BNE  loop_quad_dn
try_pair_dn:
	AND   $3, R1, R1
	LSR   $1, R1, R3
	CBZ   R3, tail_dn
loop_dn:
	FMOVD.P 8(R0), F1
	FMOVD.P 8(R2), F3
	FMULD   F3, F1, F1
	FADDD   F1, F0, F0
	FMOVD.P 8(R0), F2
	FMOVD.P 8(R2), F4
	FMULD   F4, F2, F2
	FADDD   F2, F5, F5
	SUBS $1, R3, R3
	BNE  loop_dn
tail_dn:
	FADDD F5, F0, F0
done_dn:
	FMOVD F0, ret+48(FP)
	RET

// addInPlaceNEON(dst, src []float64)
// ABI0: dst+0(FP)..16, src+24(FP)..40
TEXT ·addInPlaceNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src_len+32(FP), R3
	MOVD src+24(FP), R1
	LSR  $1, R3, R4
	CBZ  R4, done_ain
loop_ain:
	FMOVD   (R0), F0
	FMOVD.P 8(R1), F1
	FADDD   F1, F0, F0
	FMOVD.P F0, 8(R0)
	FMOVD   (R0), F2
	FMOVD.P 8(R1), F3
	FADDD   F3, F2, F2
	FMOVD.P F2, 8(R0)
	SUBS $1, R4, R4
	BNE  loop_ain
done_ain:
	RET

// mulScalarVecNEON(dst []float64, s float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP), s+24(FP)
TEXT ·mulScalarVecNEON(SB), NOSPLIT, $0-32
	MOVD  dst+0(FP), R0
	MOVD  dst_len+8(FP), R1
	FMOVD s+24(FP), F16
	LSR   $1, R1, R2
	CBZ   R2, done_msn
loop_msn:
	FMOVD   (R0), F0
	FMULD   F16, F0, F0
	FMOVD.P F0, 8(R0)
	FMOVD   (R0), F1
	FMULD   F16, F1, F1
	FMOVD.P F1, 8(R0)
	SUBS $1, R2, R2
	BNE  loop_msn
done_msn:
	RET

// reduceSumSqNEON(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceSumSqNEON(SB), NOSPLIT, $0-32
	MOVD  a+0(FP), R0
	MOVD  a_len+8(FP), R1
	FMOVD $0.0, F0
	FMOVD $0.0, F5
	LSR   $2, R1, R2
	CBZ   R2, try_pair_rssn
loop_quad_rssn:
	FMOVD.P 8(R0), F1
	FMULD   F1, F1, F1
	FADDD   F1, F0, F0
	FMOVD.P 8(R0), F2
	FMULD   F2, F2, F2
	FADDD   F2, F5, F5
	FMOVD.P 8(R0), F3
	FMULD   F3, F3, F3
	FADDD   F3, F0, F0
	FMOVD.P 8(R0), F4
	FMULD   F4, F4, F4
	FADDD   F4, F5, F5
	SUBS $1, R2, R2
	BNE  loop_quad_rssn
try_pair_rssn:
	AND   $3, R1, R1
	LSR   $1, R1, R2
	CBZ   R2, tail_rssn
loop_rssn:
	FMOVD.P 8(R0), F1
	FMULD   F1, F1, F1
	FADDD   F1, F0, F0
	FMOVD.P 8(R0), F2
	FMULD   F2, F2, F2
	FADDD   F2, F5, F5
	SUBS $1, R2, R2
	BNE  loop_rssn
tail_rssn:
	FADDD F5, F0, F0
done_rssn:
	FMOVD F0, ret+24(FP)
	RET
