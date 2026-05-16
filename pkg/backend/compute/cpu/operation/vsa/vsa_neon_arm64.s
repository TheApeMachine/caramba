#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

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
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R2), [V1.D2]
	VFMUL_D2(1, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R4, R4
	BNE  loop_bn
done_bn:
	RET

// dotReduceNEON(a, b []float64) float64
// ABI0: a+0(FP)..16, b+24(FP)..40, ret+48(FP)
TEXT ·dotReduceNEON(SB), NOSPLIT, $16-56
	MOVD  a+0(FP), R0
	MOVD  a_len+8(FP), R1
	MOVD  b+24(FP), R2
	FMOVD $0.0, F0
	VEOR V0.B16, V0.B16, V0.B16
	LSR   $1, R1, R3
	CBZ   R3, tail_dn
loop_dn:
	VLD1.P 16(R0), [V1.D2]
	VLD1.P 16(R2), [V2.D2]
	VFMUL_D2(2, 1, 1)
	VFADD_D2(1, 0, 0)
	SUBS $1, R3, R3
	BNE  loop_dn
tail_dn:
	MOVD RSP, R3
	VST1.P [V0.D2], 16(R3)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0
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
	VLD1.P 16(R0), [V0.D2]
	VLD1.P 16(R1), [V1.D2]
	VFADD_D2(1, 0, 0)
	SUB $16, R0, R0
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R4, R4
	BNE  loop_ain
done_ain:
	RET

// mulScalarVecNEON(dst []float64, s float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP), s+24(FP)
TEXT ·mulScalarVecNEON(SB), NOSPLIT, $8-32
	MOVD  dst+0(FP), R0
	MOVD  dst_len+8(FP), R1
	FMOVD s+24(FP), F16
	FMOVD F16, 0(RSP)
	VLD1R (RSP), [V16.D2]
	LSR   $1, R1, R2
	CBZ   R2, done_msn
loop_msn:
	VLD1.P 16(R0), [V0.D2]
	VFMUL_D2(16, 0, 0)
	SUB $16, R0, R0
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R2, R2
	BNE  loop_msn
done_msn:
	RET

// reduceSumSqNEON(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceSumSqNEON(SB), NOSPLIT, $16-32
	MOVD  a+0(FP), R0
	MOVD  a_len+8(FP), R1
	FMOVD $0.0, F0
	VEOR V0.B16, V0.B16, V0.B16
	LSR   $1, R1, R2
	CBZ   R2, tail_rssn
loop_rssn:
	VLD1.P 16(R0), [V1.D2]
	VFMUL_D2(1, 1, 2)
	VFADD_D2(2, 0, 0)
	SUBS $1, R2, R2
	BNE  loop_rssn
tail_rssn:
	MOVD RSP, R2
	VST1.P [V0.D2], 16(R2)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0
done_rssn:
	FMOVD F0, ret+24(FP)
	RET
