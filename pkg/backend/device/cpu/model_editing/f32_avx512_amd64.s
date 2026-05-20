// SPDX-License-Identifier: Apache-2.0
// AVX-512 weight graft: in-place weights += injection.
#include "textflag.h"

// func WeightGraftAddFloat32AVX512Asm(weights, injection *float32, count int)
TEXT ·WeightGraftAddFloat32AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ weights+0(FP), DI
	MOVQ injection+8(FP), SI
	MOVQ count+16(FP), CX

mdl_graft_w16:
	CMPQ CX, $16
	JL   mdl_graft_w8

	VMOVUPS (DI), Z0
	VMOVUPS (SI), Z1
	VADDPS  Z1, Z0, Z0
	VMOVUPS Z0, (DI)

	ADDQ $64, DI
	ADDQ $64, SI
	SUBQ $16, CX
	JMP  mdl_graft_w16

mdl_graft_w8:
	CMPQ CX, $8
	JL   mdl_graft_w4

	VMOVUPS (DI), Y0
	VMOVUPS (SI), Y1
	VADDPS  Y1, Y0, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $8, CX
	JMP  mdl_graft_w8

mdl_graft_w4:
	CMPQ CX, $4
	JL   mdl_graft_w4_tail

	VMOVUPS (DI), X0
	VMOVUPS (SI), X1
	VADDPS  X1, X0, X0
	VMOVUPS X0, (DI)

	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $4, CX
	JMP  mdl_graft_w4

mdl_graft_w4_tail:
	TESTQ CX, CX
	JZ   mdl_graft_done

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (DI), K7, Y0
	VMOVDQU32 (SI), K7, Y1
	VADDPS  Y1, Y0, Y0
	VMOVDQU32 Y0, K7, (DI)

mdl_graft_done:
	RET
