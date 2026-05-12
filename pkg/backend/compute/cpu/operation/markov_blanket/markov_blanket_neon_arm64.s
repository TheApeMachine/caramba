#include "textflag.h"

// matvecNEON(dst, w, x []float64, rows, cols int)
// dst[i] += sum_j w[i*cols+j]*x[j]; W row-major; caller copied bias into dst.
// Inner loop matches causal ·matVecNEON (2× unrolled FMADDD); see causal_neon_arm64.s.
// ABI0: dst+0(FP)..16, w+24..40, x+48..64, rows+72(FP), cols+80(FP)
TEXT ·matvecNEON(SB), NOSPLIT, $0-88
	MOVD dst+0(FP), R0
	MOVD w+24(FP), R1
	MOVD x+48(FP), R2
	MOVD rows+72(FP), R3
	MOVD cols+80(FP), R4
	CBZ  R3, done_mneon
row_loop_mneon:
	FMOVD $0.0, F0
	FMOVD $0.0, F5
	MOVD  R2, R5
	MOVD  R4, R6
	LSR   $2, R6, R7
	CBZ   R7, pair_mneon
quad_mneon:
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
	BNE  quad_mneon
pair_mneon:
	AND   $3, R6, R6
	LSR   $1, R6, R7
	CBZ   R7, scalar_mneon
	FMOVD.P 8(R1), F1
	FMOVD.P 8(R5), F2
	FMADDD  F2, F0, F1, F0
	FMOVD.P 8(R1), F3
	FMOVD.P 8(R5), F4
	FMADDD  F4, F5, F3, F5
scalar_mneon:
	AND $1, R6, R8
	CBZ R8, store_mneon
	FMOVD.P 8(R1), F1
	FMOVD.P 8(R5), F2
	FMADDD  F2, F0, F1, F0
store_mneon:
	FADDD F5, F0, F0
	FMOVD (R0), F10
	FADDD F0, F10, F10
	FMOVD F10, (R0)
	ADD $8, R0
	SUBS $1, R3, R3
	BNE row_loop_mneon
done_mneon:
	RET
