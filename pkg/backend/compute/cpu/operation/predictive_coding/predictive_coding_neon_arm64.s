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
	FMOVD $0.0, F0          // acc = 0.0
	MOVD  R2, R5            // R5 = &x[0] (reset per row)
	MOVD  R4, R6            // R6 = cols remaining
	CBZ   R6, store_mv
col_loop_mv:
	FMOVD.P 8(R1), F1       // F1 = W[i*cols+j]; R1 += 8
	FMOVD.P 8(R5), F2       // F2 = x[j]; R5 += 8
	FMULD   F2, F1, F1      // F1 = F1 * F2
	FADDD   F1, F0, F0      // F0 += F1
	SUBS $1, R6, R6
	BNE  col_loop_mv
store_mv:
	FMOVD.P F0, 8(R0)       // dst[i] = F0; R0 += 8
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
	CBZ  R6, next_tv
inner_tv:
	FMOVD.P 8(R1), F1      // F1 = W[i*cols+j]; R1 += 8
	FMOVD   (R5), F2       // F2 = dst[j]
	FMULD   F15, F1, F1    // F1 = F15 * F1
	FADDD   F1, F2, F2     // F2 += F1
	FMOVD.P F2, 8(R5)      // dst[j] = F2; R5 += 8
	SUBS $1, R6, R6
	BNE  inner_tv
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

// outerRowNEON(dst, b []float64, scale float64)
// dst[j] += scale * b[j]
TEXT ·outerRowNEON(SB), NOSPLIT, $0-56
	MOVD  dst+0(FP), R0
	MOVD  b+24(FP), R1
	MOVD  b_len+32(FP), R2
	FMOVD scale+48(FP), F16
	CBZ   R2, done_or
loop_or:
	FMOVD.P 8(R1), F0      // F0 = b[j]
	FMOVD   (R0), F1       // F1 = dst[j]
	FMULD   F16, F0, F0    // F0 = scale * b[j]
	FADDD   F0, F1, F1     // F1 = dst[j] + scale*b[j]
	FMOVD.P F1, 8(R0)
	SUBS $1, R2, R2
	BNE  loop_or
done_or:
	RET
