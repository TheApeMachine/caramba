#include "textflag.h"

// projMatmulNEON(dst, a, b []float64, M, K, N int)
// ABI0 FP layout:
//   dst+0(FP), dst_len+8(FP), dst_cap+16(FP)
//   a+24(FP),  a_len+32(FP),  a_cap+40(FP)
//   b+48(FP),  b_len+56(FP),  b_cap+64(FP)
//   M+72(FP), K+80(FP), N+88(FP)
TEXT ·projMatmulNEON(SB), NOSPLIT, $0-96
	MOVD dst+0(FP), R8
	MOVD a+24(FP), R9
	MOVD b+48(FP), R10
	MOVD M+72(FP), R0
	MOVD K+80(FP), R1
	MOVD N+88(FP), R2

	MOVD $0, R11          // i = 0
outer_i:
	CMP  R0, R11
	BGE  done_mm

	MOVD $0, R12          // j = 0
middle_j:
	CMP  R2, R12
	BGE  next_i

	FMOVD $0.0, F0        // acc = 0.0

	MOVD $0, R13          // k = 0
inner_k:
	// A[i,k]
	MUL  R1, R11, R14
	ADD  R13, R14, R14
	LSL  $3, R14, R14
	FMOVD (R9)(R14), F1

	// B[k,j]
	MUL  R2, R13, R15
	ADD  R12, R15, R15
	LSL  $3, R15, R15
	FMOVD (R10)(R15), F2

	FMADDD F1, F2, F0, F0

	ADD  $1, R13, R13
	CMP  R1, R13
	BLT  inner_k

	// C[i,j]
	MUL  R2, R11, R14
	ADD  R12, R14, R14
	LSL  $3, R14, R14
	FMOVD F0, (R8)(R14)

	ADD  $1, R12, R12
	JMP  middle_j

next_i:
	ADD  $1, R11, R11
	JMP  outer_i

done_mm:
	RET
