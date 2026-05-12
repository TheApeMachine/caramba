#include "textflag.h"

// matVecNEON(dst, W, x []float64, rows, cols int)
// ABI0: dst+0(FP), W+24(FP), x+48(FP), rows+72(FP), cols+80(FP)
// dst[i] = sum_j W[i*cols+j] * x[j]
TEXT ·matVecNEON(SB), NOSPLIT, $0-88
	MOVD dst+0(FP), R0      // R0 = &dst[0]
	MOVD W+24(FP), R1       // R1 = &W[0]
	MOVD x+48(FP), R2       // R2 = &x[0]
	MOVD rows+72(FP), R3    // R3 = rows
	MOVD cols+80(FP), R4    // R4 = cols
	CBZ  R3, done_mv
row_loop_mv:
	FMOVD $0.0, F0
	FMOVD $0.0, F5
	MOVD  R2, R5
	MOVD  R4, R6
	LSR   $2, R6, R7
	CBZ   R7, pair_mv
quad_mv:
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
	BNE  quad_mv
pair_mv:
	AND   $3, R6, R6
	LSR   $1, R6, R7
	CBZ   R7, scalar_mv
	FMOVD.P 8(R1), F1
	FMOVD.P 8(R5), F2
	FMADDD  F2, F0, F1, F0
	FMOVD.P 8(R1), F3
	FMOVD.P 8(R5), F4
	FMADDD  F4, F5, F3, F5
scalar_mv:
	AND $1, R6, R8
	CBZ R8, store_mv
	FMOVD.P 8(R1), F1
	FMOVD.P 8(R5), F2
	FMADDD  F2, F0, F1, F0
store_mv:
	FADDD F5, F0, F0
	FMOVD.P F0, 8(R0)
	SUBS $1, R3, R3
	BNE  row_loop_mv
done_mv:
	RET

// matVecTransposeNEON(dst, W, x []float64, rows, cols int)
// dst[j] = sum_i W[i*cols+j]*x[i]  — W^T @ x; dst has length cols.
TEXT ·matVecTransposeNEON(SB), NOSPLIT, $0-88
	MOVD dst+0(FP), R0
	MOVD W+24(FP), R1
	MOVD x+48(FP), R2
	MOVD rows+72(FP), R3
	MOVD cols+80(FP), R4
	// zero dst
	MOVD R0, R5
	MOVD R4, R6
zero_tv:
	CBZ  R6, zero_done_tv
	FMOVD $0.0, F0
	FMOVD.P F0, 8(R5)
	SUBS $1, R6, R6
	BNE  zero_tv
zero_done_tv:
	CBZ R3, done_tv
outer_tv:
	FMOVD.P 8(R2), F15     // F15 = x[i]; R2 += 8
	MOVD R0, R5             // R5 = &dst[0]
	MOVD R4, R6             // R6 = cols
	LSR  $2, R6, R7         // quads
	CBZ  R7, pair_tv
quad_tv:
	FMOVD.P 8(R1), F1
	FMOVD   (R5), F2
	FMULD   F15, F1, F1
	FADDD   F1, F2, F2
	FMOVD.P F2, 8(R5)
	
	FMOVD.P 8(R1), F3
	FMOVD   (R5), F4
	FMULD   F15, F3, F3
	FADDD   F3, F4, F4
	FMOVD.P F4, 8(R5)

	FMOVD.P 8(R1), F6
	FMOVD   (R5), F7
	FMULD   F15, F6, F6
	FADDD   F6, F7, F7
	FMOVD.P F7, 8(R5)

	FMOVD.P 8(R1), F8
	FMOVD   (R5), F9
	FMULD   F15, F8, F8
	FADDD   F8, F9, F9
	FMOVD.P F9, 8(R5)
	
	SUBS $1, R7, R7
	BNE  quad_tv
pair_tv:
	AND  $3, R6, R6
	LSR  $1, R6, R7
	CBZ  R7, scalar_tv
	FMOVD.P 8(R1), F1
	FMOVD   (R5), F2
	FMULD   F15, F1, F1
	FADDD   F1, F2, F2
	FMOVD.P F2, 8(R5)

	FMOVD.P 8(R1), F3
	FMOVD   (R5), F4
	FMULD   F15, F3, F3
	FADDD   F3, F4, F4
	FMOVD.P F4, 8(R5)
scalar_tv:
	AND  $1, R6, R8
	CBZ  R8, next_tv
	FMOVD.P 8(R1), F1      // F1 = W[i*cols+j]; R1 += 8
	FMOVD   (R5), F2       // F2 = dst[j]
	FMULD   F15, F1, F1    // F1 = F15 * F1
	FADDD   F1, F2, F2     // F2 += F1
	FMOVD.P F2, 8(R5)      // dst[j] = F2; R5 += 8
next_tv:
	SUBS $1, R3, R3
	BNE  outer_tv
done_tv:
	RET

// subVecNEON(dst, a, b []float64)
TEXT ·subVecNEON(SB), NOSPLIT, $0-72
	MOVD dst+0(FP), R0
	MOVD a+24(FP), R1
	MOVD b+48(FP), R2
	MOVD a_len+32(FP), R3
	CBZ  R3, done_sv
loop_sv:
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R2), F1
	FSUBD   F1, F0, F0
	FMOVD.P F0, 8(R0)
	SUBS $1, R3, R3
	BNE  loop_sv
done_sv:
	RET

// mulVecNEON(dst, a, b []float64)
TEXT ·mulVecNEON(SB), NOSPLIT, $0-72
	MOVD dst+0(FP), R0
	MOVD a+24(FP), R1
	MOVD b+48(FP), R2
	MOVD a_len+32(FP), R3
	CBZ  R3, done_mulv
loop_mulv:
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R2), F1
	FMULD   F1, F0, F0
	FMOVD.P F0, 8(R0)
	SUBS $1, R3, R3
	BNE  loop_mulv
done_mulv:
	RET

// axpyNEON(dst, src []float64, scale float64)
// dst[i] += scale * src[i]
TEXT ·axpyNEON(SB), NOSPLIT, $0-56
	MOVD  dst+0(FP), R0
	MOVD  src+24(FP), R1
	MOVD  src_len+32(FP), R2
	FMOVD scale+48(FP), F16
	CBZ   R2, done_axpy
loop_axpy:
	FMOVD.P 8(R1), F0      // F0 = src[i]
	FMOVD   (R0), F1       // F1 = dst[i]
	FMULD   F16, F0, F0    // F0 = scale * src[i]
	FADDD   F0, F1, F1     // F1 = dst[i] + scale*src[i]
	FMOVD.P F1, 8(R0)      // store and advance
	SUBS $1, R2, R2
	BNE  loop_axpy
done_axpy:
	RET
