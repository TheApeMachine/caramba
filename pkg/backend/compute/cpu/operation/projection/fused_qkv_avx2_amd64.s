#include "textflag.h"

// fusedQKVMatmulAVX2(dst, input, weight []float64, M, K, N int)
// ABI0 FP layout:
//   dst+0(FP)   ptr, dst_len+8(FP), dst_cap+16(FP)
//   a+24(FP)    ptr, a_len+32(FP),  a_cap+40(FP)
//   b+48(FP)    ptr, b_len+56(FP),  b_cap+64(FP)
//   M+72(FP), K+80(FP), N+88(FP)
//
// Tiled 4x4 AVX2 matmul using VFMADD231PD.
// QKV[i,j] = sum_k x[i,k]*W[k,j], row-major for projection.fused_qkv.
TEXT ·fusedQKVMatmulAVX2(SB), NOSPLIT, $0-96
	MOVQ dst+0(FP), R8
	MOVQ a+24(FP), R9
	MOVQ b+48(FP), R10
	MOVQ M+72(FP), R11   // M
	MOVQ K+80(FP), R12   // K
	MOVQ N+88(FP), R13   // N

	XORQ R14, R14        // i = 0
outer_i:
	CMPQ R14, R11
	JGE  done_mm

	XORQ R15, R15        // j = 0
middle_j:
	MOVQ R13, CX
	SUBQ $4, CX
	CMPQ R15, CX
	JG   tail_j

	VXORPD Y0, Y0, Y0   // acc = 0

	XORQ AX, AX          // k = 0
inner_k_avx:
	// x[i,k]
	MOVQ R14, BX
	IMULQ R12, BX
	ADDQ AX, BX
	VMOVSD (R9)(BX*8), X15
	VBROADCASTSD X15, Y15

	// W[k, j..j+3]
	MOVQ AX, BX
	IMULQ R13, BX
	ADDQ R15, BX
	VMOVUPD (R10)(BX*8), Y1

	VFMADD231PD Y15, Y1, Y0

	INCQ AX
	CMPQ AX, R12
	JL   inner_k_avx

	// QKV[i, j..j+3]
	MOVQ R14, BX
	IMULQ R13, BX
	ADDQ R15, BX
	VMOVUPD Y0, (R8)(BX*8)

	ADDQ $4, R15
	JMP  middle_j

tail_j:
	CMPQ R15, R13
	JGE  next_i
	VXORPD X0, X0, X0
	XORQ AX, AX
inner_k_scalar:
	MOVQ R14, BX
	IMULQ R12, BX
	ADDQ AX, BX
	VMOVSD (R9)(BX*8), X1
	MOVQ AX, BX
	IMULQ R13, BX
	ADDQ R15, BX
	VMOVSD (R10)(BX*8), X2
	VMULSD X2, X1, X1
	VADDSD X1, X0, X0
	INCQ AX
	CMPQ AX, R12
	JL   inner_k_scalar
	MOVQ R14, BX
	IMULQ R13, BX
	ADDQ R15, BX
	VMOVSD X0, (R8)(BX*8)
	INCQ R15
	JMP  tail_j

next_i:
	INCQ R14
	JMP  outer_i

done_mm:
	VZEROUPPER
	RET
