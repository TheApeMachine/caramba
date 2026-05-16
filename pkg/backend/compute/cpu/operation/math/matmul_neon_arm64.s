#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

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

	CBZ  R0, mm_done
	CBZ  R2, mm_done

	MOVD ZR, R11         // i = 0
mm_outer:
	CMP  R0, R11
	BGE  mm_done

	MOVD ZR, R12         // j = 0
mm_middle:
	ADD  $1, R12, R3
	CMP  R2, R3
	BGE  mm_tail_j

	VEOR V0.B16, V0.B16, V0.B16
	MOVD ZR, R13         // k = 0
mm_inner:
	MUL  R1, R11, R14
	ADD  R13, R14
	LSL  $3, R14
	ADD  R9, R14, R14
	VLD1R (R14), [V15.D2]

	MUL  R2, R13, R15
	ADD  R12, R15
	LSL  $3, R15
	ADD  R10, R15, R15
	VLD1.P 16(R15), [V1.D2]

	VFMUL_D2(15, 1, 2)
	VFADD_D2(2, 0, 0)

	ADD  $1, R13
	CMP  R1, R13
	BLT  mm_inner

	MUL  R2, R11, R14
	ADD  R12, R14
	LSL  $3, R14
	ADD  R8, R14, R14
	VST1.P [V0.D2], 16(R14)

	ADD  $2, R12
	B    mm_middle

mm_tail_j:
	CMP  R2, R12
	BGE  mm_next_i

	FMOVD ZR, F0
	MOVD ZR, R13
mm_inner_tail:
	MUL  R1, R11, R14
	ADD  R13, R14
	LSL  $3, R14
	FMOVD (R9)(R14), F1

	MUL  R2, R13, R15
	ADD  R12, R15
	LSL  $3, R15
	FMOVD (R10)(R15), F2

	FMULD F1, F2, F3
	FADDD F3, F0, F0

	ADD  $1, R13
	CMP  R1, R13
	BLT  mm_inner_tail

	MUL  R2, R11, R14
	ADD  R12, R14
	LSL  $3, R14
	FMOVD F0, (R8)(R14)

	ADD  $1, R12
	B    mm_tail_j

mm_next_i:
	ADD  $1, R11
	B    mm_outer

mm_done:
	RET
