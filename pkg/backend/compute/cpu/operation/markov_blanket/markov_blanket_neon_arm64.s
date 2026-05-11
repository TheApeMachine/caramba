#include "textflag.h"

// matvecNEON(dst, w, x []float64, rows, cols int)
// dst[i] += sum_j( w[i*cols+j] * x[j] )
// ABI0 layout:
//   dst+0(FP)  ptr, +8 len, +16 cap
//   w+24(FP)   ptr, +32 len, +40 cap
//   x+48(FP)   ptr, +56 len, +64 cap
//   rows+72(FP)
//   cols+80(FP)
TEXT ·matvecNEON(SB), NOSPLIT, $0-88
	MOVD dst+0(FP), R0
	MOVD w+24(FP), R1
	MOVD x+48(FP), R2
	MOVD rows+72(FP), R3
	MOVD cols+80(FP), R4
	CBZ  R3, done_mneon
row_loop_mneon:
	FMOVD $0.0, F0
	MOVD  R4, R5        // col counter
	MOVD  R1, R6        // row-start in w
	MOVD  R2, R7        // x ptr (reset)
	LSR   $1, R5, R8
	CBZ   R8, tail_mneon
vec_loop_mneon:
	FMOVD.P 8(R6), F1
	FMOVD.P 8(R7), F2
	FMADDD  F2, F0, F1, F0
	FMOVD.P 8(R6), F3
	FMOVD.P 8(R7), F4
	FMADDD  F4, F0, F3, F0
	SUBS $1, R8, R8
	BNE  vec_loop_mneon
tail_mneon:
	// handle odd remaining column
	AND  $1, R5, R9
	CBZ  R9, write_mneon
	FMOVD.P 8(R6), F1
	FMOVD.P 8(R7), F2
	FMADDD  F2, F0, F1, F0
write_mneon:
	FMOVD   (R0), F5
	FADDD   F0, F5, F5
	FMOVD.P F5, 8(R0)
	// advance w ptr by cols*8 bytes
	LSL  $3, R4, R9
	ADD  R9, R1, R1
	SUBS $1, R3, R3
	BNE  row_loop_mneon
done_mneon:
	RET
