// SPDX-License-Identifier: Apache-2.0
// AVX-512 float32 shape kernels: contiguous copy, where, masked fill.
#include "textflag.h"

// func CopyContiguousFloat32AVX512Asm(dst, src *float32, count int)
TEXT ·CopyContiguousFloat32AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX

shape_copy_w16:
	CMPQ CX, $16
	JL   shape_copy_w8

	VMOVUPS (SI), Z0
	VMOVUPS Z0, (DI)
	VMOVUPS 64(SI), Z1
	VMOVUPS Z1, 64(DI)

	ADDQ $128, SI
	ADDQ $128, DI
	SUBQ $16, CX
	JMP  shape_copy_w16

shape_copy_w8:
	CMPQ CX, $8
	JL   shape_copy_w4

	VMOVUPS (SI), Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  shape_copy_w8

shape_copy_w4:
	CMPQ CX, $4
	JL   shape_copy_w4_tail

	VMOVUPS (SI), X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  shape_copy_w4

shape_copy_w4_tail:
	TESTQ CX, CX
	JZ   shape_copy_done

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y0
	VMOVDQU32 Y0, K7, (DI)

shape_copy_done:
	RET

// func WhereFloat32AVX512Asm(dst, positive, negative *float32, mask *byte, count int)
TEXT ·WhereFloat32AVX512Asm(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), DI
	MOVQ positive+8(FP), SI
	MOVQ negative+16(FP), R8
	MOVQ mask+24(FP), R9
	MOVQ count+32(FP), CX
	XORQ R11, R11

shape_where_w16:
	CMPQ CX, $16
	JL   shape_where_w8

	CMPQ R11, $0
	JNE  shape_where_w8

	MOVW (R9), AX
	KMOVW AX, K7

	VMOVUPS (SI), Z0
	VMOVUPS (R8), Z1
	VBLENDMPS Z0, Z1, K7, Z0
	VMOVUPS Z0, (DI)

	ADDQ $64, SI
	ADDQ $64, R8
	ADDQ $64, DI
	ADDQ $2, R9
	SUBQ $16, CX
	JMP  shape_where_w16

shape_where_w8:
	CMPQ CX, $8
	JL   shape_where_tail

shape_where_w8_load:
	CMPQ R11, $0
	JNE  shape_where_w8_use

	MOVBQZX (R9), R10

shape_where_w8_use:
	MOVQ  R10, AX
	MOVQ  R11, BX
	MOVQ  BX, CX
	SHRQ  CL, AX
	ANDQ  $0xFF, AX
	KMOVQ AX, K7

	VMOVUPS (SI), Y0
	VMOVUPS (R8), Y1
	VBLENDMPS Y0, Y1, K7, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, DI
	ADDQ $8, R11
	CMPQ R11, $8
	JL   shape_where_w8_no_advance

	XORQ R11, R11
	ADDQ $1, R9

shape_where_w8_no_advance:
	SUBQ $8, CX
	JMP  shape_where_w8

shape_where_tail:
	TESTQ CX, CX
	JZ   shape_where_done

shape_where_tail_load:
	CMPQ R11, $0
	JNE  shape_where_tail_mask

	MOVBQZX (R9), R10

shape_where_tail_mask:
	MOVQ  CX, DX
	MOVQ  R10, AX
	MOVQ  R11, BX
	MOVQ  BX, CX
	SHRQ  CL, AX
	MOVQ  DX, CX
	MOVQ  $1, R12
	SHLQ  CL, R12
	DECQ  R12
	ANDQ  R12, AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y0
	VMOVDQU32 (R8), K7, Y1
	VBLENDMPS Y0, Y1, K7, Y0
	VMOVDQU32 Y0, K7, (DI)

shape_where_done:
	RET

// func MaskedFillFloat32AVX512Asm(dst, input *float32, fill float32, mask *byte, count int)
TEXT ·MaskedFillFloat32AVX512Asm(SB), NOSPLIT, $0-36
	MOVQ dst+0(FP), DI
	MOVQ input+8(FP), SI
	MOVSS fill+16(FP), X0
	MOVQ mask+24(FP), R9
	MOVQ count+32(FP), CX
	XORQ R11, R11

	VBROADCASTSS X0, Z31

shape_mfill_w16:
	CMPQ CX, $16
	JL   shape_mfill_w8

	CMPQ R11, $0
	JNE  shape_mfill_w8

	MOVW (R9), AX
	KMOVW AX, K7

	VMOVUPS (SI), Z0
	VBLENDMPS Z31, Z0, K7, Z0
	VMOVUPS Z0, (DI)

	ADDQ $64, SI
	ADDQ $64, DI
	ADDQ $2, R9
	SUBQ $16, CX
	JMP  shape_mfill_w16

shape_mfill_w8:
	CMPQ CX, $8
	JL   shape_mfill_tail

shape_mfill_w8_load:
	CMPQ R11, $0
	JNE  shape_mfill_w8_use

	MOVBQZX (R9), R10

shape_mfill_w8_use:
	MOVQ  R10, AX
	MOVQ  R11, BX
	MOVQ  BX, CX
	SHRQ  CL, AX
	ANDQ  $0xFF, AX
	KMOVQ AX, K7

	VMOVUPS (SI), Y0
	VBLENDMPS Y31, Y0, K7, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $8, R11
	CMPQ R11, $8
	JL   shape_mfill_w8_no_advance

	XORQ R11, R11
	ADDQ $1, R9

shape_mfill_w8_no_advance:
	SUBQ $8, CX
	JMP  shape_mfill_w8

shape_mfill_tail:
	TESTQ CX, CX
	JZ   shape_mfill_done

shape_mfill_tail_load:
	CMPQ R11, $0
	JNE  shape_mfill_tail_mask

	MOVBQZX (R9), R10

shape_mfill_tail_mask:
	MOVQ  CX, DX
	MOVQ  R10, AX
	MOVQ  R11, BX
	MOVQ  BX, CX
	SHRQ  CL, AX
	MOVQ  DX, CX
	MOVQ  $1, R12
	SHLQ  CL, R12
	DECQ  R12
	ANDQ  R12, AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y0
	VBLENDMPS Y31, Y0, K7, Y0
	VMOVDQU32 Y0, K7, (DI)

shape_mfill_done:
	RET
