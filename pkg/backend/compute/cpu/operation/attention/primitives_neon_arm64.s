#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFDIV_D2(m, n, d) WORD $(0x6E60FC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFMAXNM_D2(m, n, d) WORD $(0x4E60C400 | ((m) << 16) | ((n) << 5) | (d))

DATA ·primZero+0(SB)/8, $0.0
GLOBL ·primZero(SB), RODATA|NOPTR, $8

// dotProductNEON(a, b []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP),
//       b+24(FP), b_len+32(FP), b_cap+40(FP),
//       ret+48(FP)
TEXT ·dotProductNEON(SB), NOSPLIT, $16-56
	MOVD  a+0(FP), R0
	MOVD  a_len+8(FP), R2
	MOVD  b+24(FP), R1
	VEOR  V0.B16, V0.B16, V0.B16
	LSR   $1, R2, R3
	CBZ   R3, dp_neon_tail
dp_neon_loop:
	VLD1.P 16(R0), [V1.D2]
	VLD1.P 16(R1), [V2.D2]
	VFMUL_D2(2, 1, 3)
	VFADD_D2(3, 0, 0)
	SUBS $1, R3, R3
	BNE  dp_neon_loop
	MOVD RSP, R4
	VST1.P [V0.D2], 16(R4)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0
dp_neon_tail:
	TST $1, R2
	BEQ dp_neon_done
	FMOVD.P 8(R0), F1
	FMOVD.P 8(R1), F2
	FMADDD F1, F0, F2, F0
dp_neon_done:
	FMOVD F0, ret+48(FP)
	RET

// scaledAddNEON(dst, src []float64, scale float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP),
//       src+24(FP), src_len+32(FP), src_cap+40(FP),
//       scale+48(FP)
TEXT ·scaledAddNEON(SB), NOSPLIT, $8-56
	MOVD  dst+0(FP), R0
	MOVD  dst_len+8(FP), R2
	MOVD  src+24(FP), R1
	FMOVD scale+48(FP), F16
	FMOVD F16, 0(RSP)
	VLD1R (RSP), [V16.D2]
	LSR   $1, R2, R3
	CBZ   R3, sa_neon_tail
sa_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1   (R0), [V1.D2]
	VFMUL_D2(16, 0, 0)
	VFADD_D2(0, 1, 1)
	VST1.P [V1.D2], 16(R0)
	SUBS $1, R3, R3
	BNE  sa_neon_loop
sa_neon_tail:
	TST $1, R2
	BEQ sa_neon_done
	FMOVD.P 8(R1), F0
	FMOVD   (R0), F2
	FMULD   F16, F0, F0
	FADDD   F0, F2, F2
	FMOVD.P F2, 8(R0)
sa_neon_done:
	RET

// reduceMaxNEON(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceMaxNEON(SB), NOSPLIT, $16-32
	MOVD  a+0(FP), R0
	MOVD  a_len+8(FP), R1
	FMOVD ·primZero(SB), F0
	CBZ   R1, rm_neon_done
	CMP   $2, R1
	BLT   rm_neon_scalar_first
	VLD1.P 16(R0), [V0.D2]
	SUB  $2, R1, R1
	LSR  $1, R1, R2
	CBZ  R2, rm_neon_horizontal
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

// reduceSumNEON(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceSumNEON(SB), NOSPLIT, $16-32
	MOVD  a+0(FP), R0
	MOVD  a_len+8(FP), R1
	VEOR  V0.B16, V0.B16, V0.B16
	LSR   $1, R1, R2
	CBZ   R2, rs_neon_tail
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

// divScalarNEON(dst []float64, s float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP), s+24(FP)
TEXT ·divScalarNEON(SB), NOSPLIT, $8-32
	MOVD  dst+0(FP), R0
	MOVD  dst_len+8(FP), R1
	FMOVD s+24(FP), F16
	FMOVD F16, 0(RSP)
	VLD1R (RSP), [V16.D2]
	LSR   $1, R1, R2
	CBZ   R2, ds_neon_tail
ds_neon_loop:
	VLD1 (R0), [V0.D2]
	VFDIV_D2(16, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R2, R2
	BNE  ds_neon_loop
ds_neon_tail:
	TST $1, R1
	BEQ ds_neon_done
	FMOVD   (R0), F0
	FDIVD   F16, F0, F0
	FMOVD.P F0, 8(R0)
ds_neon_done:
	RET
