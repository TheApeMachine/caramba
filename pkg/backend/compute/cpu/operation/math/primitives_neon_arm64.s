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
// dst[i] /= s
TEXT ·divScalarNEON(SB), NOSPLIT, $0-32
	MOVD   dst+0(FP), R0
	MOVD   dst_len+8(FP), R1
	FMOVD  s+24(FP), F16
	LSR    $1, R1, R2
	CBZ    R2, done_ds
loop_ds:
	FMOVD   (R0), F0
	FDIVD   F16, F0, F0
	FMOVD.P F0, 8(R0)
	FMOVD   (R0), F1
	FDIVD   F16, F1, F1
	FMOVD.P F1, 8(R0)
	SUBS $1, R2, R2
	BNE  loop_ds
done_ds:
	RET

// addVecNEON(dst, a, b []float64)
// ABI0: dst+0(FP)..16, a+24(FP)..40, b+48(FP)..64
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
// dst[i] *= s  — uses non-post-increment load to avoid pointer aliasing bug
TEXT ·mulScalarNEON(SB), NOSPLIT, $0-32
	MOVD   dst+0(FP), R0
	MOVD   dst_len+8(FP), R1
	FMOVD  s+24(FP), F16
	LSR    $1, R1, R2
	CBZ    R2, done_ms
loop_ms:
	FMOVD   (R0), F0
	FMULD   F16, F0, F0
	FMOVD.P F0, 8(R0)
	FMOVD   (R0), F1
	FMULD   F16, F1, F1
	FMOVD.P F1, 8(R0)
	SUBS $1, R2, R2
	BNE  loop_ms
done_ms:
	RET

// reduceSumSqNEON(a []float64) float64
// Computes sum of a[i]^2
// FMADDD Fm, Fa, Fn, Fd  ⟹  Fd = Fa + Fn*Fm
// To compute F0 += F1*F1: FMADDD F1, F0, F1, F0
TEXT ·reduceSumSqNEON(SB), NOSPLIT, $0-32
	MOVD   a+0(FP), R0
	MOVD   a_len+8(FP), R1
	FMOVD  $0.0, F0
	LSR    $1, R1, R2
	CBZ    R2, done_ssq
loop_ssq:
	FMOVD.P 8(R0), F1
	FMADDD  F1, F0, F1, F0
	FMOVD.P 8(R0), F2
	FMADDD  F2, F0, F2, F0
	SUBS $1, R2, R2
	BNE  loop_ssq
done_ssq:
	FMOVD F0, ret+24(FP)
	RET

// signVecNEON(dst, src []float64)
// ABI0: dst+0(FP)..16, src+24(FP)..40
TEXT ·signVecNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD src_len+32(FP), R2
	FMOVD $0.0, F31
	FMOVD $1.0, F29
	FMOVD $-1.0, F30
	CBZ  R2, done_sv
loop_sv:
	FMOVD.P 8(R1), F0
	FCMPD   F31, F0
	BEQ  zero_sv
	BMI  neg_sv
	FMOVD.P F29, 8(R0)
	B    next_sv
neg_sv:
	FMOVD.P F30, 8(R0)
	B    next_sv
zero_sv:
	FMOVD.P F31, 8(R0)
next_sv:
	SUBS $1, R2, R2
	BNE  loop_sv
done_sv:
	RET

// outerRowNEON(dst, b []float64, scale float64)
// ABI0: dst+0(FP)..16, b+24(FP)..40, scale+48(FP)
TEXT ·outerRowNEON(SB), NOSPLIT, $0-56
	MOVD   dst+0(FP), R0
	MOVD   b+24(FP), R1
	MOVD   b_len+32(FP), R2
	FMOVD  scale+48(FP), F15
	LSR    $1, R2, R3
	CBZ    R3, done_or
loop_or:
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R1), F1
	FMULD   F15, F0, F0
	FMULD   F15, F1, F1
	FMOVD.P F0, 8(R0)
	FMOVD.P F1, 8(R0)
	SUBS $1, R3, R3
	BNE  loop_or
done_or:
	RET

// addScaledVecNEON(dst, src []float64, scale float64)
// dst[i] += scale * src[i]
// ABI0: dst+0(FP)..16, src+24(FP)..40, scale+48(FP)
TEXT ·addScaledVecNEON(SB), NOSPLIT, $0-56
	MOVD  dst+0(FP), R0
	MOVD  src+24(FP), R1
	MOVD  src_len+32(FP), R2
	FMOVD scale+48(FP), F15
	LSR   $1, R2, R3
	CBZ   R3, done_asv
loop_asv:
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R1), F1
	FMULD   F15, F0, F2
	FMULD   F15, F1, F3
	FMOVD   (R0), F4
	FADDD   F2, F4, F4
	FMOVD.P F4, 8(R0)
	FMOVD   (R0), F5
	FADDD   F3, F5, F5
	FMOVD.P F5, 8(R0)
	SUBS $1, R3, R3
	BNE  loop_asv
done_asv:
	RET

// sqrtVecNEON(dst, src []float64)
// ABI0: dst+0(FP)..16, src+24(FP)..40
TEXT ·sqrtVecNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD src_len+32(FP), R2
	LSR  $1, R2, R3
	CBZ  R3, done_sqv
loop_sqv:
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R1), F1
	FSQRTD  F0, F0
	FSQRTD  F1, F1
	FMOVD.P F0, 8(R0)
	FMOVD.P F1, 8(R0)
	SUBS $1, R3, R3
	BNE  loop_sqv
done_sqv:
	RET

// addScalarVecNEON(dst []float64, scalar float64)
// dst[i] += scalar  — use non-post-increment load then post-increment store
// ABI0: dst+0(FP)..16, scalar+24(FP)
TEXT ·addScalarVecNEON(SB), NOSPLIT, $0-32
	MOVD  dst+0(FP), R0
	MOVD  dst_len+8(FP), R1
	FMOVD scalar+24(FP), F15
	LSR   $1, R1, R2
	CBZ   R2, done_asa
loop_asa:
	FMOVD   (R0), F0
	FADDD   F15, F0, F0
	FMOVD.P F0, 8(R0)
	FMOVD   (R0), F1
	FADDD   F15, F1, F1
	FMOVD.P F1, 8(R0)
	SUBS $1, R2, R2
	BNE  loop_asa
done_asa:
	RET

// divVecNEON(dst, a, b []float64)
// dst[i] = a[i] / b[i]
// ABI0: dst+0(FP)..16, a+24(FP)..40, b+48(FP)..64
TEXT ·divVecNEON(SB), NOSPLIT, $0-72
	MOVD dst+0(FP), R0
	MOVD a+24(FP), R1
	MOVD a_len+32(FP), R2
	MOVD b+48(FP), R3
	LSR  $1, R2, R4
	CBZ  R4, done_dv
loop_dv:
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R1), F1
	FMOVD.P 8(R3), F2
	FMOVD.P 8(R3), F3
	FDIVD   F2, F0, F0
	FDIVD   F3, F1, F1
	FMOVD.P F0, 8(R0)
	FMOVD.P F1, 8(R0)
	SUBS $1, R4, R4
	BNE  loop_dv
done_dv:
	RET

// l2NormSqNEON(a []float64) float64
// Returns sum(a[i]^2)
// FMADDD Fm, Fa, Fn, Fd  ⟹  Fd = Fa + Fn*Fm
// To compute F0 += F1*F1: FMADDD F1, F0, F1, F0
// ABI0: a+0(FP)..16, ret+24(FP)
TEXT ·l2NormSqNEON(SB), NOSPLIT, $0-32
	MOVD  a+0(FP), R0
	MOVD  a_len+8(FP), R1
	FMOVD $0.0, F0
	LSR   $1, R1, R2
	CBZ   R2, done_l2
loop_l2:
	FMOVD.P 8(R0), F1
	FMADDD  F1, F0, F1, F0
	FMOVD.P 8(R0), F2
	FMADDD  F2, F0, F2, F0
	SUBS $1, R2, R2
	BNE  loop_l2
done_l2:
	FMOVD F0, ret+24(FP)
	RET

// clampVecNEON(dst []float64, lo, hi float64)
// dst[i] = clamp(dst[i], lo, hi)
// ABI0: dst+0(FP)..16, lo+24(FP), hi+32(FP)
TEXT ·clampVecNEON(SB), NOSPLIT, $0-40
	MOVD  dst+0(FP), R0
	MOVD  dst_len+8(FP), R1
	FMOVD lo+24(FP), F14
	FMOVD hi+32(FP), F15
	LSR   $1, R1, R2
	CBZ   R2, done_cv
loop_cv:
	FMOVD   (R0), F0
	FMAXD   F14, F0, F0
	FMIND   F15, F0, F0
	FMOVD.P F0, 8(R0)
	FMOVD   (R0), F1
	FMAXD   F14, F1, F1
	FMIND   F15, F1, F1
	FMOVD.P F1, 8(R0)
	SUBS $1, R2, R2
	BNE  loop_cv
done_cv:
	RET
