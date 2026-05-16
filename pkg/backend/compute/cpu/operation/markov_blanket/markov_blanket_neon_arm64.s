#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// matvecNEON(dst, w, x []float64, rows, cols int)
// dst[i] += sum_j w[i*cols+j]*x[j]; W row-major; caller copied bias into dst.
// ABI0: dst+0(FP)..16, w+24..40, x+48..64, rows+72(FP), cols+80(FP)
TEXT ·matvecNEON(SB), NOSPLIT, $16-88
	MOVD dst+0(FP), R0
	MOVD w+24(FP), R1
	MOVD x+48(FP), R2
	MOVD rows+72(FP), R3
	MOVD cols+80(FP), R4
	CBZ  R3, done_mneon
row_loop_mneon:
	FMOVD $0.0, F0
	MOVD  R2, R5
	MOVD  R4, R6
	VEOR  V0.B16, V0.B16, V0.B16
	LSR   $1, R6, R7
	CBZ   R7, scalar_mneon
pair_mneon:
	VLD1.P 16(R1), [V1.D2]
	VLD1.P 16(R5), [V2.D2]
	VFMUL_D2(1, 2, 3)
	VFADD_D2(3, 0, 0)
	SUBS $1, R7, R7
	BNE  pair_mneon

	MOVD RSP, R7
	VST1.P [V0.D2], 16(R7)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0

scalar_mneon:
	AND $1, R6, R8
	CBZ R8, store_mneon
	FMOVD.P 8(R1), F1
	FMOVD.P 8(R5), F2
	FMADDD  F2, F0, F1, F0
store_mneon:
	FMOVD (R0), F10
	FADDD F0, F10, F10
	FMOVD F10, (R0)
	ADD $8, R0
	SUBS $1, R3, R3
	BNE row_loop_mneon
done_mneon:
	RET
