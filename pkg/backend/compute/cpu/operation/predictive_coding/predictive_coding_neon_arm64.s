#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// matVecNEON(dst, W, x []float64, rows, cols int)
// ABI0: dst+0(FP), W+24(FP), x+48(FP), rows+72(FP), cols+80(FP)
// dst[i] = sum_j W[i*cols+j] * x[j]
TEXT ·matVecNEON(SB), NOSPLIT, $16-88
	MOVD dst+0(FP), R0      // R0 = &dst[0]
	MOVD W+24(FP), R1       // R1 = &W[0]
	MOVD x+48(FP), R2       // R2 = &x[0]
	MOVD rows+72(FP), R3    // R3 = rows
	MOVD cols+80(FP), R4    // R4 = cols
	CBZ  R3, done_mv
row_loop_mv:
	FMOVD $0.0, F0
	MOVD  R2, R5
	MOVD  R4, R6
	VEOR  V0.B16, V0.B16, V0.B16
	LSR   $1, R6, R7
	CBZ   R7, scalar_mv
pair_mv:
	VLD1.P 16(R1), [V1.D2]
	VLD1.P 16(R5), [V2.D2]
	VFMUL_D2(1, 2, 3)
	VFADD_D2(3, 0, 0)
	SUBS $1, R7, R7
	BNE  pair_mv

	MOVD RSP, R7
	VST1.P [V0.D2], 16(R7)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0

scalar_mv:
	AND $1, R6, R8
	CBZ R8, store_mv
	FMOVD.P 8(R1), F1
	FMOVD.P 8(R5), F2
	FMADDD  F2, F0, F1, F0
store_mv:
	FMOVD.P F0, 8(R0)
	SUBS $1, R3, R3
	BNE  row_loop_mv
done_mv:
	RET

// matVecTransposeNEON(dst, W, x []float64, rows, cols int)
// dst[j] = sum_i W[i*cols+j]*x[i]  — W^T @ x; dst has length cols.
TEXT ·matVecTransposeNEON(SB), NOSPLIT, $8-88
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
	VEOR V0.B16, V0.B16, V0.B16
	LSR  $1, R6, R7
	CBZ  R7, zero_tail_tv
zero_pair_tv:
	VST1.P [V0.D2], 16(R5)
	SUBS $1, R7, R7
	BNE  zero_pair_tv
zero_tail_tv:
	TST  $1, R6
	BEQ  zero_done_tv
	FMOVD $0.0, F0
	FMOVD F0, (R5)
zero_done_tv:
	CBZ R3, done_tv
outer_tv:
	FMOVD.P 8(R2), F15     // F15 = x[i]; R2 += 8
	FMOVD F15, 0(RSP)
	VLD1R (RSP), [V15.D2]
	MOVD R0, R5             // R5 = &dst[0]
	MOVD R4, R6             // R6 = cols
	LSR  $1, R6, R7
	CBZ  R7, scalar_tv
pair_tv:
	VLD1.P 16(R1), [V1.D2]
	VLD1.P 16(R5), [V2.D2]
	VFMUL_D2(15, 1, 1)
	VFADD_D2(1, 2, 2)
	SUB $16, R5, R5
	VST1.P [V2.D2], 16(R5)
	SUBS $1, R7, R7
	BNE  pair_tv

scalar_tv:
	TST  $1, R6
	BEQ  next_tv
	FMOVD.P 8(R1), F1
	FMOVD   (R5), F2
	FMULD   F15, F1, F1
	FADDD   F1, F2, F2
	FMOVD.P F2, 8(R5)
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
	LSR  $1, R3, R4
	CBZ  R4, tail_sv
loop_sv:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R2), [V1.D2]
	VFSUB_D2(1, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R4, R4
	BNE  loop_sv

tail_sv:
	TST $1, R3
	BEQ done_sv
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R2), F1
	FSUBD   F1, F0, F0
	FMOVD.P F0, 8(R0)
done_sv:
	RET

// mulVecNEON(dst, a, b []float64)
TEXT ·mulVecNEON(SB), NOSPLIT, $0-72
	MOVD dst+0(FP), R0
	MOVD a+24(FP), R1
	MOVD b+48(FP), R2
	MOVD a_len+32(FP), R3
	CBZ  R3, done_mulv
	LSR  $1, R3, R4
	CBZ  R4, tail_mulv
loop_mulv:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R2), [V1.D2]
	VFMUL_D2(1, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R4, R4
	BNE  loop_mulv

tail_mulv:
	TST $1, R3
	BEQ done_mulv
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R2), F1
	FMULD   F1, F0, F0
	FMOVD.P F0, 8(R0)
done_mulv:
	RET

// axpyNEON(dst, src []float64, scale float64)
// dst[i] += scale * src[i]
TEXT ·axpyNEON(SB), NOSPLIT, $8-56
	MOVD  dst+0(FP), R0
	MOVD  src+24(FP), R1
	MOVD  src_len+32(FP), R2
	FMOVD scale+48(FP), F16
	FMOVD F16, 0(RSP)
	VLD1R (RSP), [V16.D2]
	CBZ   R2, done_axpy
	LSR   $1, R2, R3
	CBZ   R3, tail_axpy
loop_axpy:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R0), [V1.D2]
	VFMUL_D2(16, 0, 0)
	VFADD_D2(0, 1, 1)
	SUB $16, R0, R0
	VST1.P [V1.D2], 16(R0)
	SUBS $1, R3, R3
	BNE  loop_axpy

tail_axpy:
	TST $1, R2
	BEQ done_axpy
	FMOVD.P 8(R1), F0      // F0 = src[i]
	FMOVD   (R0), F1       // F1 = dst[i]
	FMULD   F16, F0, F0    // F0 = scale * src[i]
	FADDD   F0, F1, F1     // F1 = dst[i] + scale*src[i]
	FMOVD.P F1, 8(R0)      // store and advance
done_axpy:
	RET
