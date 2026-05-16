#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFDIV_D2(m, n, d) WORD $(0x6E60FC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFSQRT_D2(n, d) WORD $(0x6EE1F800 | ((n) << 5) | (d))
#define VFMINNM_D2(m, n, d) WORD $(0x4EE0C400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMAXNM_D2(m, n, d) WORD $(0x4E60C400 | ((m) << 16) | ((n) << 5) | (d))
#define VFCMGT_D2(m, n, d) WORD $(0x6EE0E400 | ((m) << 16) | ((n) << 5) | (d))

// reduceSumNEON(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceSumNEON(SB), NOSPLIT, $16-32
	MOVD a+0(FP), R0
	MOVD a_len+8(FP), R1
	VEOR V0.B16, V0.B16, V0.B16
	LSR  $1, R1, R2
	CBZ  R2, rs_neon_tail
rs_neon_loop:
	VLD1.P 16(R0), [V1.D2]
	VFADD_D2(1, 0, 0)
	SUBS $1, R2, R2
	BNE  rs_neon_loop
	MOVD RSP, R2
	VST1.P [V0.D2], 16(R2)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0
rs_neon_tail:
	TST $1, R1
	BEQ rs_neon_done
	FMOVD.P 8(R0), F1
	FADDD F1, F0, F0
rs_neon_done:
	FMOVD F0, ret+24(FP)
	RET

// reduceMaxNEON(a []float64) float64
TEXT ·reduceMaxNEON(SB), NOSPLIT, $16-32
	MOVD a+0(FP), R0
	MOVD a_len+8(FP), R1
	MOVD $0xFFEFFFFFFFFFFFFF, R2
	FMOVD R2, F0
	CBZ  R1, rm_neon_done
	CMP  $2, R1
	BLT  rm_neon_scalar_first
	VLD1.P 16(R0), [V0.D2]
	SUB $2, R1, R1
	LSR $1, R1, R2
	CBZ R2, rm_neon_horizontal
rm_neon_loop:
	VLD1.P 16(R0), [V1.D2]
	VFMAXNM_D2(1, 0, 0)
	SUBS $1, R2, R2
	BNE  rm_neon_loop
rm_neon_horizontal:
	MOVD RSP, R2
	VST1.P [V0.D2], 16(R2)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FMAXD F1, F0, F0
	TST $1, R1
	BEQ rm_neon_done
	FMOVD.P 8(R0), F1
	FMAXD F1, F0, F0
	B rm_neon_done
rm_neon_scalar_first:
	FMOVD.P 8(R0), F0
rm_neon_done:
	FMOVD F0, ret+24(FP)
	RET

// divScalarNEON(dst []float64, s float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP), s+24(FP)
// dst[i] /= s
TEXT ·divScalarNEON(SB), NOSPLIT, $8-32
	MOVD  dst+0(FP), R0
	MOVD  dst_len+8(FP), R1
	FMOVD s+24(FP), F16
	FMOVD F16, 0(RSP)
	VLD1R (RSP), [V16.D2]
	LSR   $1, R1, R2
	CBZ   R2, ds_neon_done
ds_neon_loop:
	VLD1 (R0), [V0.D2]
	VFDIV_D2(16, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R2, R2
	BNE  ds_neon_loop
ds_neon_done:
	RET

// addVecNEON(dst, a, b []float64)
// ABI0: dst+0(FP)..16, a+24(FP)..40, b+48(FP)..64
TEXT ·addVecNEON(SB), NOSPLIT, $0-72
	MOVD dst+0(FP), R0
	MOVD a_len+32(FP), R3
	MOVD a+24(FP), R1
	MOVD b+48(FP), R2
	LSR  $1, R3, R4
	CBZ  R4, av_neon_done
av_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R2), [V1.D2]
	VFADD_D2(1, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R4, R4
	BNE  av_neon_loop
av_neon_done:
	RET

// mulVecNEON(dst, a, b []float64)
TEXT ·mulVecNEON(SB), NOSPLIT, $0-72
	MOVD dst+0(FP), R0
	MOVD a_len+32(FP), R3
	MOVD a+24(FP), R1
	MOVD b+48(FP), R2
	LSR  $1, R3, R4
	CBZ  R4, mv_neon_done
mv_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R2), [V1.D2]
	VFMUL_D2(1, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R4, R4
	BNE  mv_neon_loop
mv_neon_done:
	RET

// mulScalarNEON(dst []float64, s float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP), s+24(FP)
TEXT ·mulScalarNEON(SB), NOSPLIT, $8-32
	MOVD  dst+0(FP), R0
	MOVD  dst_len+8(FP), R1
	FMOVD s+24(FP), F16
	FMOVD F16, 0(RSP)
	VLD1R (RSP), [V16.D2]
	LSR   $1, R1, R2
	CBZ   R2, ms_neon_done
ms_neon_loop:
	VLD1 (R0), [V0.D2]
	VFMUL_D2(16, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R2, R2
	BNE  ms_neon_loop
ms_neon_done:
	RET

// reduceSumSqNEON(a []float64) float64
// Computes the even vector prefix; Go owns the odd tail.
TEXT ·reduceSumSqNEON(SB), NOSPLIT, $16-32
	MOVD a+0(FP), R0
	MOVD a_len+8(FP), R1
	VEOR V0.B16, V0.B16, V0.B16
	LSR  $1, R1, R2
	CBZ  R2, ssq_neon_done
ssq_neon_loop:
	VLD1.P 16(R0), [V1.D2]
	VFMUL_D2(1, 1, 1)
	VFADD_D2(1, 0, 0)
	SUBS $1, R2, R2
	BNE  ssq_neon_loop
	MOVD RSP, R2
	VST1.P [V0.D2], 16(R2)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0
ssq_neon_done:
	FMOVD F0, ret+24(FP)
	RET

// signVecNEON(dst, src []float64)
// ABI0: dst+0(FP)..16, src+24(FP)..40
TEXT ·signVecNEON(SB), NOSPLIT, $16-48
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD src_len+32(FP), R2
	FMOVD $1.0, F29
	FMOVD F29, 0(RSP)
	VLD1R (RSP), [V29.D2]
	FMOVD $-1.0, F30
	FMOVD F30, 8(RSP)
	ADD $8, RSP, R3
	VLD1R (R3), [V30.D2]
	VEOR V31.B16, V31.B16, V31.B16
	LSR $1, R2, R3
	CBZ R3, sv_neon_done
sv_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VFCMGT_D2(31, 0, 1)
	VFCMGT_D2(0, 31, 2)
	VAND V29.B16, V1.B16, V1.B16
	VAND V30.B16, V2.B16, V2.B16
	VORR  V2.B16, V1.B16, V1.B16
	VST1.P [V1.D2], 16(R0)
	SUBS $1, R3, R3
	BNE  sv_neon_loop
sv_neon_done:
	RET

// outerRowNEON(dst, b []float64, scale float64)
// ABI0: dst+0(FP)..16, b+24(FP)..40, scale+48(FP)
TEXT ·outerRowNEON(SB), NOSPLIT, $8-56
	MOVD  dst+0(FP), R0
	MOVD  b+24(FP), R1
	MOVD  b_len+32(FP), R2
	FMOVD scale+48(FP), F15
	FMOVD F15, 0(RSP)
	VLD1R (RSP), [V15.D2]
	LSR   $1, R2, R3
	CBZ   R3, or_neon_done
or_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VFMUL_D2(15, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R3, R3
	BNE  or_neon_loop
or_neon_done:
	RET

// addScaledVecNEON(dst, src []float64, scale float64)
// dst[i] += scale * src[i]
// ABI0: dst+0(FP)..16, src+24(FP)..40, scale+48(FP)
TEXT ·addScaledVecNEON(SB), NOSPLIT, $8-56
	MOVD  dst+0(FP), R0
	MOVD  src+24(FP), R1
	MOVD  src_len+32(FP), R2
	FMOVD scale+48(FP), F15
	FMOVD F15, 0(RSP)
	VLD1R (RSP), [V15.D2]
	LSR   $1, R2, R3
	CBZ   R3, asv_neon_done
asv_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1   (R0), [V1.D2]
	VFMUL_D2(15, 0, 0)
	VFADD_D2(0, 1, 1)
	VST1.P [V1.D2], 16(R0)
	SUBS $1, R3, R3
	BNE  asv_neon_loop
asv_neon_done:
	RET

// sqrtVecNEON(dst, src []float64)
// ABI0: dst+0(FP)..16, src+24(FP)..40
TEXT ·sqrtVecNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD src_len+32(FP), R2
	LSR  $1, R2, R3
	CBZ  R3, sqv_neon_done
sqv_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VFSQRT_D2(0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R3, R3
	BNE  sqv_neon_loop
sqv_neon_done:
	RET

// addScalarVecNEON(dst []float64, scalar float64)
// ABI0: dst+0(FP)..16, scalar+24(FP)
TEXT ·addScalarVecNEON(SB), NOSPLIT, $8-32
	MOVD  dst+0(FP), R0
	MOVD  dst_len+8(FP), R1
	FMOVD scalar+24(FP), F15
	FMOVD F15, 0(RSP)
	VLD1R (RSP), [V15.D2]
	LSR   $1, R1, R2
	CBZ   R2, asa_neon_done
asa_neon_loop:
	VLD1 (R0), [V0.D2]
	VFADD_D2(15, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R2, R2
	BNE  asa_neon_loop
asa_neon_done:
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
	CBZ  R4, dv_neon_done
dv_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R3), [V1.D2]
	VFDIV_D2(1, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R4, R4
	BNE  dv_neon_loop
dv_neon_done:
	RET

// clampVecNEON(dst []float64, lo, hi float64)
// dst[i] = clamp(dst[i], lo, hi)
// ABI0: dst+0(FP)..16, lo+24(FP), hi+32(FP)
TEXT ·clampVecNEON(SB), NOSPLIT, $16-40
	MOVD  dst+0(FP), R0
	MOVD  dst_len+8(FP), R1
	FMOVD lo+24(FP), F14
	FMOVD F14, 0(RSP)
	VLD1R (RSP), [V14.D2]
	FMOVD hi+32(FP), F15
	FMOVD F15, 8(RSP)
	ADD $8, RSP, R3
	VLD1R (R3), [V15.D2]
	LSR $1, R1, R2
	CBZ R2, cv_neon_done
cv_neon_loop:
	VLD1 (R0), [V0.D2]
	VFMAXNM_D2(14, 0, 0)
	VFMINNM_D2(15, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R2, R2
	BNE  cv_neon_loop
cv_neon_done:
	RET
