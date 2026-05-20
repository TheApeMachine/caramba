// SPDX-License-Identifier: Apache-2.0
// AVX-512 int32 tokenizer pack: contiguous copy (tokenizer_pack_int32).
#include "textflag.h"

// func TokenizerPackInt32AVX512Asm(dst, src *int32, count int)
TEXT ·TokenizerPackInt32AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX

tok_pack_w16:
	CMPQ CX, $16
	JL   tok_pack_w8

	VMOVDQU32 (SI), Z0
	VMOVDQU32 Z0, (DI)

	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $16, CX
	JMP  tok_pack_w16

tok_pack_w8:
	CMPQ CX, $8
	JL   tok_pack_w4

	VMOVDQU32 (SI), Y0
	VMOVDQU32 Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  tok_pack_w8

tok_pack_w4:
	CMPQ CX, $4
	JL   tok_pack_w4_tail

	VMOVDQU32 (SI), X0
	VMOVDQU32 X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  tok_pack_w4

tok_pack_w4_tail:
	TESTQ CX, CX
	JZ   tok_pack_done

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y0
	VMOVDQU32 Y0, K7, (DI)

tok_pack_done:
	RET
