#include "textflag.h"

// Go's arm64 assembler does not accept every FP64 NEON mnemonic directly.
// These encodings match the activation package's two-lane float64 primitives.
#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// fusedQKVMatmulNEON(dst, input, weight []float64, M, K, N int)
// ABI0 FP layout:
//   dst+0(FP), dst_len+8(FP), dst_cap+16(FP)
//   a+24(FP),  a_len+32(FP),  a_cap+40(FP)
//   b+48(FP),  b_len+56(FP),  b_cap+64(FP)
//   M+72(FP), K+80(FP), N+88(FP)
TEXT ·fusedQKVMatmulNEON(SB), NOSPLIT, $0-96
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
	ADD  $1, R12, R3
	CMP  R2, R3
	BGE  tail_j

	VEOR V0.B16, V0.B16, V0.B16
	MOVD $0, R13          // k = 0
inner_k_neon:
	MUL  R1, R11, R14
	ADD  R13, R14, R14
	LSL  $3, R14, R14
	ADD  R9, R14, R14
	VLD1R (R14), [V15.D2]

	MUL  R2, R13, R15
	ADD  R12, R15, R15
	LSL  $3, R15, R15
	ADD  R10, R15, R15
	VLD1.P 16(R15), [V1.D2]

	VFMUL_D2(15, 1, 2)
	VFADD_D2(2, 0, 0)

	ADD  $1, R13, R13
	CMP  R1, R13
	BLT  inner_k_neon

	MUL  R2, R11, R14
	ADD  R12, R14, R14
	LSL  $3, R14, R14
	ADD  R8, R14, R14
	VST1.P [V0.D2], 16(R14)

	ADD  $2, R12, R12
	JMP  middle_j

tail_j:
	CMP  R2, R12
	BGE  next_i

	FMOVD $0.0, F0
	MOVD  $0, R13
inner_k_tail:
	MUL  R1, R11, R14
	ADD  R13, R14, R14
	LSL  $3, R14, R14
	FMOVD (R9)(R14), F1

	MUL  R2, R13, R15
	ADD  R12, R15, R15
	LSL  $3, R15, R15
	FMOVD (R10)(R15), F2

	FMADDD F1, F0, F2, F0

	ADD  $1, R13, R13
	CMP  R1, R13
	BLT  inner_k_tail

	MUL  R2, R11, R14
	ADD  R12, R14, R14
	LSL  $3, R14, R14
	FMOVD F0, (R8)(R14)

	ADD  $1, R12, R12
	JMP  tail_j

next_i:
	ADD  $1, R11, R11
	JMP  outer_i

done_mm:
	RET
