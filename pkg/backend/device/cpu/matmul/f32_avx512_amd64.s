#include "textflag.h"

// func MatmulRowFloat32AVX512Asm(cRow, aRow, b *float32, inner, cols int)
//
// C[row, :] += A[row, :] × B for row-major B[inner, cols].
// Hot loops broadcast A[row, k] and FMA against contiguous B[k, j..j+W].
TEXT ·MatmulRowFloat32AVX512Asm(SB), NOSPLIT, $0-40
	MOVQ cRow+0(FP), DI
	MOVQ aRow+8(FP), SI
	MOVQ b+16(FP), BX
	MOVQ inner+24(FP), R12
	MOVQ cols+32(FP), DX

	MOVQ DX, R8
	SHLQ $2, R8
	MOVQ R8, R10

mm_col_loop:
	TESTQ DX, DX
	JZ    mm_done

	CMPQ DX, $16
	JL    mm_col_w8

	VMOVUPS Z0, (DI)
	MOVQ  R12, CX
	MOVQ  SI, R13
	MOVQ  BX, R14

mm_k16:
	TESTQ CX, CX
	JZ    mm_k16_done

	MOVSS (R13), X1
	VBROADCASTSS X1, Z1
	VMOVUPS (R14), Z2
	VFMADD231PS Z0, Z2, Z1
	ADDQ  $4, R13
	ADDQ  R10, R14
	DECQ  CX
	JMP   mm_k16

mm_k16_done:
	VMOVUPS Z0, (DI)
	ADDQ  $64, DI
	ADDQ  $64, BX
	SUBQ  $16, DX
	JMP   mm_col_loop

mm_col_w8:
	TESTQ DX, DX
	JZ    mm_done

	CMPQ DX, $8
	JL    mm_col_w4

	VMOVUPS Y0, (DI)
	MOVQ  R12, CX
	MOVQ  SI, R13
	MOVQ  BX, R14

mm_k8:
	TESTQ CX, CX
	JZ    mm_k8_done

	MOVSS (R13), X1
	VBROADCASTSS X1, Y1
	VMOVUPS (R14), Y2
	VFMADD231PS Y0, Y2, Y1
	ADDQ  $4, R13
	ADDQ  R10, R14
	DECQ  CX
	JMP   mm_k8

mm_k8_done:
	VMOVUPS Y0, (DI)
	ADDQ  $32, DI
	ADDQ  $32, BX
	SUBQ  $8, DX
	JMP   mm_col_loop

mm_col_w4:
	TESTQ DX, DX
	JZ    mm_done

	CMPQ DX, $4
	JL    mm_col_w4_tail

	VMOVUPS X0, (DI)
	MOVQ  R12, CX
	MOVQ  SI, R13
	MOVQ  BX, R14

mm_k4:
	TESTQ CX, CX
	JZ    mm_k4_done

	MOVSS (R13), X1
	VBROADCASTSS X1, X2
	VMOVUPS (R14), X3
	VFMADD231PS X0, X3, X2
	ADDQ  $4, R13
	ADDQ  R10, R14
	DECQ  CX
	JMP   mm_k4

mm_k4_done:
	VMOVUPS X0, (DI)
	ADDQ  $16, DI
	ADDQ  $16, BX
	SUBQ  $4, DX
	JMP   mm_col_loop

mm_col_w4_tail:
	TESTQ DX, DX
	JZ    mm_done

	MOVQ  DX, CX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (DI), K7, Y0
	MOVQ  R12, CX
	MOVQ  SI, R13
	MOVQ  BX, R14

mm_k_tail:
	TESTQ CX, CX
	JZ    mm_k_tail_done

	MOVSS (R13), X1
	VBROADCASTSS X1, Y1
	VMOVDQU32 (R14), K7, Y2
	VFMADD231PS Y0, Y2, Y1
	ADDQ  $4, R13
	ADDQ  R10, R14
	DECQ  CX
	JMP   mm_k_tail

mm_k_tail_done:
	VMOVDQU32 Y0, K7, (DI)
	XORQ  DX, DX
	JMP   mm_col_loop

mm_done:
	RET
