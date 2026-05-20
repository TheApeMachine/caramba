// SPDX-License-Identifier: Apache-2.0
// AVX-512 activation steering: dst = base + coefficient * direction.
#include "textflag.h"

// func ActivationSteerFloat32AVX512Asm(dst, base, direction *float32, coefficient float32, count int)
TEXT ·ActivationSteerFloat32AVX512Asm(SB), NOSPLIT, $0-36
	MOVQ dst+0(FP), DI
	MOVQ base+8(FP), SI
	MOVQ direction+16(FP), R8
	MOVSS coefficient+24(FP), X15
	VBROADCASTSS X15, Z15
	MOVQ count+32(FP), CX

intrp_steer_w16:
	CMPQ CX, $16
	JL   intrp_steer_w8

	VMOVUPS (SI), Z0
	VMOVUPS (R8), Z1
	VFMADD231PS Z15, Z1, Z0
	VMOVUPS Z0, (DI)

	ADDQ $64, SI
	ADDQ $64, R8
	ADDQ $64, DI
	SUBQ $16, CX
	JMP  intrp_steer_w16

intrp_steer_w8:
	CMPQ CX, $8
	JL   intrp_steer_w4

	VMOVUPS (SI), Y0
	VMOVUPS (R8), Y1
	VFMADD231PS Y15, Y1, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  intrp_steer_w8

intrp_steer_w4:
	CMPQ CX, $4
	JL   intrp_steer_w4_tail

	VMOVUPS (SI), X0
	VMOVUPS (R8), X1
	VFMADD231PS X15, X1, X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  intrp_steer_w4

intrp_steer_w4_tail:
	TESTQ CX, CX
	JZ   intrp_steer_done

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y0
	VMOVDQU32 (R8), K7, Y1
	VFMADD231PS Y15, Y1, Y0
	VMOVDQU32 Y0, K7, (DI)

intrp_steer_done:
	RET
