#include "textflag.h"

DATA ·mmZero+0(SB)/8, $0.0
GLOBL ·mmZero(SB), RODATA|NOPTR, $8

// matmulNEON(dst, a, b []float64, M, K, N int)
// ABI0 FP layout:
//   dst+0(FP), dst_len+8(FP), dst_cap+16(FP)
//   a+24(FP),  a_len+32(FP),  a_cap+40(FP)
//   b+48(FP),  b_len+56(FP),  b_cap+64(FP)
//   M+72(FP), K+80(FP), N+88(FP)
TEXT ·matmulNEON(SB), NOSPLIT, $0-96
	MOVD dst+0(FP), R8
	MOVD a+24(FP),  R9
	MOVD b+48(FP),  R10
	MOVD M+72(FP),  R0   // M
	MOVD K+80(FP),  R1   // K
	MOVD N+88(FP),  R2   // N

	FMOVD ·mmZero(SB), F30  // constant 0.0

	CBZ  R0, mm_done

	MOVD ZR, R11         // i = 0
mm_outer:
	CMP  R0, R11
	BGE  mm_done

	MOVD ZR, R12         // j = 0
mm_middle:
	CMP  R2, R12
	BGE  mm_next_i

	FMOVD F30, F0        // acc = 0.0

	MOVD ZR, R13         // k = 0
mm_inner:
	// A[i,k]: index = i*K + k
	MUL  R1, R11, R14
	ADD  R13, R14
	LSL  $3, R14
	FMOVD (R9)(R14), F1

	// B[k,j]: index = k*N + j
	MUL  R2, R13, R15
	ADD  R12, R15
	LSL  $3, R15
	FMOVD (R10)(R15), F2

	FMADDD F1, F2, F0, F0   // F0 = F0 + F2*F1

	ADD  $1, R13
	CMP  R1, R13
	BLT  mm_inner

	// C[i,j]: index = i*N + j
	MUL  R2, R11, R14
	ADD  R12, R14
	LSL  $3, R14
	FMOVD F0, (R8)(R14)

	ADD  $1, R12
	B    mm_middle

mm_next_i:
	ADD  $1, R11
	B    mm_outer

mm_done:
	RET
