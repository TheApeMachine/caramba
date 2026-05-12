#include "textflag.h"

// matVecNEON(dst, w, x []float64, rows, cols int)
// dst = W @ x  where W is [rows x cols] row-major.
// ABI0: dst+0..16, w+24..40, x+48..64, rows+72, cols+80
TEXT ·matVecNEON(SB), NOSPLIT, $0-88
	MOVD dst+0(FP), R0
	MOVD w+24(FP), R1
	MOVD x+48(FP), R2
	MOVD rows+72(FP), R3
	MOVD cols+80(FP), R4
	CBZ  R3, done_mv_neon
row_loop_mv_neon:
	FMOVD $0.0, F0            // accumulator
	FMOVD $0.0, F5            // accumulator 2
	MOVD  R2, R5              // x pointer (reset per row)
	MOVD  R4, R6              // cols remaining
	LSR   $2, R6, R7
	CBZ   R7, pair_mv_neon
quad_mv_neon:
	FMOVD.P 8(R1), F1
	FMOVD.P 8(R5), F2
	FMADDD  F2, F0, F1, F0
	FMOVD.P 8(R1), F3
	FMOVD.P 8(R5), F4
	FMADDD  F4, F5, F3, F5
	FMOVD.P 8(R1), F8
	FMOVD.P 8(R5), F9
	FMADDD  F9, F0, F8, F0
	FMOVD.P 8(R1), F10
	FMOVD.P 8(R5), F11
	FMADDD  F11, F5, F10, F5
	SUBS $1, R7, R7
	BNE  quad_mv_neon
pair_mv_neon:
	AND   $3, R6, R6
	LSR   $1, R6, R7
	CBZ   R7, scalar_mv_neon
	FMOVD.P 8(R1), F1
	FMOVD.P 8(R5), F2
	FMADDD  F2, F0, F1, F0
	FMOVD.P 8(R1), F3
	FMOVD.P 8(R5), F4
	FMADDD  F4, F5, F3, F5
scalar_mv_neon:
	AND $1, R6, R8
	CBZ R8, store_mv_neon
	FMOVD.P 8(R1), F1
	FMOVD.P 8(R5), F2
	FMADDD  F2, F0, F1, F0
store_mv_neon:
	FADDD F5, F0, F0
	FMOVD.P F0, 8(R0)
	SUBS $1, R3, R3
	BNE  row_loop_mv_neon
done_mv_neon:
	RET

// axpyNEON(dst, src []float64, scale float64)
// dst[i] += scale * src[i]
// ABI0: dst+0..16, src+24..40, scale+48
TEXT ·axpyNEON(SB), NOSPLIT, $0-56
	MOVD  dst+0(FP), R0
	MOVD  src_len+32(FP), R1
	MOVD  src+24(FP), R2
	MOVD  dst+0(FP), R6       // dst base
	MOVD  src+24(FP), R7      // src base
	MOVD  src_len+32(FP), R8  // original len
	FMOVD scale+48(FP), F15
	LSR   $1, R1, R3
	CBZ   R3, tail_axpy_neon
loop_axpy_neon:
	FMOVD.P 8(R2), F0
	FMOVD.P 8(R2), F1
	FMULD   F15, F0, F2
	FMULD   F15, F1, F3
	FMOVD   (R0), F4
	FADDD   F2, F4, F4
	FMOVD.P F4, 8(R0)
	FMOVD   (R0), F5
	FADDD   F3, F5, F5
	FMOVD.P F5, 8(R0)
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
TEXT ·dotNEON(SB), NOSPLIT, $0-56
	MOVD  a+0(FP), R4       // a base
	MOVD  a_len+8(FP), R6   // len
	MOVD  b+24(FP), R5      // b base
	FMOVD $0.0, F0
	FMOVD $0.0, F5
	MOVD  R4, R0
	MOVD  R5, R2
	MOVD  R6, R3
	LSR   $2, R3, R10       // quads
	CBZ   R10, dot_pair_neon
loop_dot_quad_neon:
	FMOVD.P 8(R0), F1
	FMOVD.P 8(R2), F3
	FMADDD  F3, F0, F1, F0
	FMOVD.P 8(R0), F2
	FMOVD.P 8(R2), F4
	FMADDD  F4, F5, F2, F5
	FMOVD.P 8(R0), F8
	FMOVD.P 8(R2), F9
	FMADDD  F9, F0, F8, F0
	FMOVD.P 8(R0), F10
	FMOVD.P 8(R2), F11
	FMADDD  F11, F5, F10, F5
	SUBS $1, R10, R10
	BNE  loop_dot_quad_neon
dot_pair_neon:
	AND   $3, R3, R3
	LSR   $1, R3, R10
	CBZ   R10, dot_tail_neon
loop_dot_pair_neon:
	FMOVD.P 8(R0), F1
	FMOVD.P 8(R2), F3
	FMADDD  F3, F0, F1, F0
	FMOVD.P 8(R0), F2
	FMOVD.P 8(R2), F4
	FMADDD  F4, F5, F2, F5
	SUBS $1, R10, R10
	BNE  loop_dot_pair_neon
dot_tail_neon:
	FADDD F5, F0, F0
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
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R1), F1
	FMOVD.P 8(R3), F2
	FMOVD.P 8(R3), F3
	FSUBD   F2, F0, F0
	FSUBD   F3, F1, F1
	FMOVD.P F0, 8(R0)
	FMOVD.P F1, 8(R0)
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
