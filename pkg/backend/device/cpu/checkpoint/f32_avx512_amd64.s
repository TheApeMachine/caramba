// SPDX-License-Identifier: Apache-2.0
// AVX-512 checkpoint f32 payload: little-endian float32 bytes copy (encode/decode data).
#include "textflag.h"

// func CheckpointEncodeFloat32DataAVX512Asm(dst *byte, src *float32, count int)
TEXT ·CheckpointEncodeFloat32DataAVX512Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX

ckpt_enc_w16:
	CMPQ CX, $16
	JL   ckpt_enc_w8

	VMOVDQU32 (SI), Z0
	VMOVDQU32 Z0, (DI)

	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $16, CX
	JMP  ckpt_enc_w16

ckpt_enc_w8:
	CMPQ CX, $8
	JL   ckpt_enc_w4

	VMOVDQU32 (SI), Y0
	VMOVDQU32 Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  ckpt_enc_w8

ckpt_enc_w4:
	CMPQ CX, $4
	JL   ckpt_enc_w4_tail

	VMOVDQU32 (SI), X0
	VMOVDQU32 X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  ckpt_enc_w4

ckpt_enc_w4_tail:
	TESTQ CX, CX
	JZ   ckpt_enc_done

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y0
	VMOVDQU32 Y0, K7, (DI)

ckpt_enc_done:
	RET

// func CheckpointDecodeFloat32DataAVX512Asm(dst *float32, src *byte, count int)
TEXT ·CheckpointDecodeFloat32DataAVX512Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX

ckpt_dec_w16:
	CMPQ CX, $16
	JL   ckpt_dec_w8

	VMOVDQU32 (SI), Z0
	VMOVDQU32 Z0, (DI)

	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $16, CX
	JMP  ckpt_dec_w16

ckpt_dec_w8:
	CMPQ CX, $8
	JL   ckpt_dec_w4

	VMOVDQU32 (SI), Y0
	VMOVDQU32 Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  ckpt_dec_w8

ckpt_dec_w4:
	CMPQ CX, $4
	JL   ckpt_dec_w4_tail

	VMOVDQU32 (SI), X0
	VMOVDQU32 X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  ckpt_dec_w4

ckpt_dec_w4_tail:
	TESTQ CX, CX
	JZ   ckpt_dec_done

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y0
	VMOVDQU32 Y0, K7, (DI)

ckpt_dec_done:
	RET
