#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// matVecNEON(dst, w, x []float64, rows, cols int)
// dst = W @ x  where W is [rows x cols] row-major.
// ABI0: dst+0..16, w+24..40, x+48..64, rows+72, cols+80
TEXT ·matVecNEON(SB), NOSPLIT, $16-88
	MOVD dst+0(FP), R0
	MOVD w+24(FP), R1
	MOVD x+48(FP), R2
	MOVD rows+72(FP), R3
	MOVD cols+80(FP), R4
	CBZ  R3, done_mv_neon
row_loop_mv_neon:
	FMOVD $0.0, F0            // accumulator
	MOVD  R2, R5              // x pointer (reset per row)
	MOVD  R4, R6              // cols remaining
	VEOR  V0.B16, V0.B16, V0.B16
	LSR   $1, R6, R7
	CBZ   R7, scalar_mv_neon
pair_mv_neon:
	VLD1.P 16(R1), [V1.D2]
	VLD1.P 16(R5), [V2.D2]
	VFMUL_D2(1, 2, 3)
	VFADD_D2(3, 0, 0)
	SUBS $1, R7, R7
	BNE  pair_mv_neon

	MOVD RSP, R7
	VST1.P [V0.D2], 16(R7)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0

scalar_mv_neon:
	AND $1, R6, R8
	CBZ R8, store_mv_neon
	FMOVD.P 8(R1), F1
	FMOVD.P 8(R5), F2
	FMADDD  F2, F0, F1, F0
store_mv_neon:
	FMOVD.P F0, 8(R0)
	SUBS $1, R3, R3
	BNE  row_loop_mv_neon
done_mv_neon:
	RET

// axpyNEON(dst, src []float64, scale float64)
// dst[i] += scale * src[i]
// ABI0: dst+0..16, src+24..40, scale+48
TEXT ·axpyNEON(SB), NOSPLIT, $8-56
	MOVD  dst+0(FP), R0
	MOVD  src_len+32(FP), R1
	MOVD  src+24(FP), R2
	MOVD  dst+0(FP), R6       // dst base
	MOVD  src+24(FP), R7      // src base
	MOVD  src_len+32(FP), R8  // original len
	FMOVD scale+48(FP), F15
	FMOVD F15, 0(RSP)
	VLD1R (RSP), [V15.D2]
	LSR   $1, R1, R3
	CBZ   R3, tail_axpy_neon
loop_axpy_neon:
	VLD1.P 16(R2), [V0.D2]
	VLD1.P 16(R0), [V1.D2]
	VFMUL_D2(15, 0, 0)
	VFADD_D2(0, 1, 1)
	SUB $16, R0, R0
	VST1.P [V1.D2], 16(R0)
	SUBS $1, R3, R3
	BNE  loop_axpy_neon
tail_axpy_neon:
	TST  $1, R8
	BEQ  done_axpy_neon
	SUB  $1, R8, R9
	ADD  R9<<3, R6, R10
	ADD  R9<<3, R7, R11
	FMOVD (R11), F0
	FMULD F15, F0, F1
	FMOVD (R10), F2
	FADDD F1, F2, F2
	FMOVD F2, (R10)
done_axpy_neon:
	RET

// dotNEON(a, b []float64) float64
// ABI0: a+0..16, b+24..40, ret+48
TEXT ·dotNEON(SB), NOSPLIT, $16-56
	MOVD  a+0(FP), R4       // a base
	MOVD  a_len+8(FP), R6   // len
	MOVD  b+24(FP), R5      // b base
	FMOVD $0.0, F0
	VEOR V0.B16, V0.B16, V0.B16
	MOVD  R4, R0
	MOVD  R5, R2
	MOVD  R6, R3
	LSR   $1, R3, R10
	CBZ   R10, dot_tail_neon
loop_dot_pair_neon:
	VLD1.P 16(R0), [V1.D2]
	VLD1.P 16(R2), [V2.D2]
	VFMUL_D2(2, 1, 1)
	VFADD_D2(1, 0, 0)
	SUBS $1, R10, R10
	BNE  loop_dot_pair_neon
dot_tail_neon:
	MOVD RSP, R10
	VST1.P [V0.D2], 16(R10)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0
	TST  $1, R6
	BEQ  done_dot_neon
	SUB  $1, R6, R7
	ADD  R7<<3, R4, R8
	ADD  R7<<3, R5, R9
	FMOVD (R8), F1
	FMOVD (R9), F2
	FMADDD F2, F0, F1, F0
done_dot_neon:
	FMOVD F0, ret+48(FP)
	RET

// subVecNEON(dst, a, b []float64)
// dst[i] = a[i] - b[i]
// ABI0: dst+0..16, a+24..40, b+48..64
TEXT ·subVecNEON(SB), NOSPLIT, $0-72
	MOVD dst+0(FP), R0
	MOVD a+24(FP), R1
	MOVD a_len+32(FP), R2
	MOVD b+48(FP), R3
	MOVD dst+0(FP), R6      // dst base
	MOVD a+24(FP), R7       // a base
	MOVD b+48(FP), R8       // b base
	MOVD a_len+32(FP), R12 // len
	LSR  $1, R2, R4
	CBZ  R4, tail_sub_neon
loop_sub_neon:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R3), [V1.D2]
	VFSUB_D2(1, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R4, R4
	BNE  loop_sub_neon
tail_sub_neon:
	TST  $1, R12
	BEQ  done_sub_neon
	SUB  $1, R12, R9
	ADD  R9<<3, R7, R10
	ADD  R9<<3, R8, R11
	ADD  R9<<3, R6, R13
	FMOVD (R10), F0
	FMOVD (R11), F1
	FSUBD F1, F0, F0
	FMOVD F0, (R13)
done_sub_neon:
	RET
