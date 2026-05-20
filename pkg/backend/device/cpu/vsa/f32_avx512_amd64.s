// SPDX-License-Identifier: Apache-2.0
// AVX-512 float32 VSA kernels: bind (mul), bundle (add), permute copy, similarity (dot).
#include "textflag.h"

// func VsaBindFloat32AVX512Asm(dst, left, right *float32, count int)
TEXT ·VsaBindFloat32AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ count+24(FP), CX

vsa_bind_w16:
	CMPQ CX, $16
	JL   vsa_bind_w8

	VMOVUPS (SI), Z0
	VMOVUPS (R8), Z1
	VMULPS  Z1, Z0, Z0
	VMOVUPS Z0, (DI)

	ADDQ $64, SI
	ADDQ $64, R8
	ADDQ $64, DI
	SUBQ $16, CX
	JMP  vsa_bind_w16

vsa_bind_w8:
	CMPQ CX, $8
	JL   vsa_bind_w4

	VMOVUPS (SI), Y0
	VMOVUPS (R8), Y1
	VMULPS  Y1, Y0, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  vsa_bind_w8

vsa_bind_w4:
	CMPQ CX, $4
	JL   vsa_bind_w4_tail

	VMOVUPS (SI), X0
	VMOVUPS (R8), X1
	VMULPS  X1, X0, X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  vsa_bind_w4

vsa_bind_w4_tail:
	TESTQ CX, CX
	JZ   vsa_bind_done

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y0
	VMOVDQU32 (R8), K7, Y1
	VMULPS  Y1, Y0, Y0
	VMOVDQU32 Y0, K7, (DI)

vsa_bind_done:
	RET

// func VsaBundleFloat32AVX512Asm(dst, left, right *float32, count int)
TEXT ·VsaBundleFloat32AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ left+8(FP), SI
	MOVQ right+16(FP), R8
	MOVQ count+24(FP), CX

vsa_bundle_w16:
	CMPQ CX, $16
	JL   vsa_bundle_w8

	VMOVUPS (SI), Z0
	VMOVUPS (R8), Z1
	VADDPS  Z1, Z0, Z0
	VMOVUPS Z0, (DI)

	ADDQ $64, SI
	ADDQ $64, R8
	ADDQ $64, DI
	SUBQ $16, CX
	JMP  vsa_bundle_w16

vsa_bundle_w8:
	CMPQ CX, $8
	JL   vsa_bundle_w4

	VMOVUPS (SI), Y0
	VMOVUPS (R8), Y1
	VADDPS  Y1, Y0, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  vsa_bundle_w8

vsa_bundle_w4:
	CMPQ CX, $4
	JL   vsa_bundle_w4_tail

	VMOVUPS (SI), X0
	VMOVUPS (R8), X1
	VADDPS  X1, X0, X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, R8
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  vsa_bundle_w4

vsa_bundle_w4_tail:
	TESTQ CX, CX
	JZ   vsa_bundle_done

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y0
	VMOVDQU32 (R8), K7, Y1
	VADDPS  Y1, Y0, Y0
	VMOVDQU32 Y0, K7, (DI)

vsa_bundle_done:
	RET

// func VsaPermuteCopyFloat32AVX512Asm(dst, src *float32, count int)
TEXT ·VsaPermuteCopyFloat32AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX

vsa_copy_w16:
	CMPQ CX, $16
	JL   vsa_copy_w8

	VMOVUPS (SI), Z0
	VMOVUPS Z0, (DI)

	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $16, CX
	JMP  vsa_copy_w16

vsa_copy_w8:
	CMPQ CX, $8
	JL   vsa_copy_w4

	VMOVUPS (SI), Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  vsa_copy_w8

vsa_copy_w4:
	CMPQ CX, $4
	JL   vsa_copy_w4_tail

	VMOVUPS (SI), X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  vsa_copy_w4

vsa_copy_w4_tail:
	TESTQ CX, CX
	JZ   vsa_copy_done

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y0
	VMOVDQU32 Y0, K7, (DI)

vsa_copy_done:
	RET

// func VsaSimilarityFloat32AVX512Asm(left, right *float32, count int) float32
TEXT ·VsaSimilarityFloat32AVX512Asm(SB), NOSPLIT, $0-28
	MOVQ left+0(FP), SI
	MOVQ right+8(FP), DI
	MOVQ count+16(FP), CX

	TESTQ CX, CX
	JZ   vsa_sim_zero

	VXORPD Y0, Y0, Y0

vsa_sim_w8:
	CMPQ CX, $8
	JL   vsa_sim_w4

	VMOVUPS (SI), Y1
	VMOVUPS (DI), Y2
	VEXTRACTF128 $0, Y1, X3
	VEXTRACTF128 $0, Y2, X4
	VCVTPS2PD X3, Y5
	VCVTPS2PD X4, Y6
	VFMADD231PD Y0, Y6, Y5
	VEXTRACTF128 $1, Y1, X3
	VEXTRACTF128 $1, Y2, X4
	VCVTPS2PD X3, Y5
	VCVTPS2PD X4, Y6
	VFMADD231PD Y0, Y6, Y5

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  vsa_sim_w8

vsa_sim_w4:
	CMPQ CX, $4
	JL   vsa_sim_w4_tail

	VMOVUPS (SI), X1
	VMOVUPS (DI), X2
	VCVTPS2PD X1, Y5
	VCVTPS2PD X2, Y6
	VFMADD231PD Y0, Y6, Y5

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  vsa_sim_w4

vsa_sim_w4_tail:
	TESTQ CX, CX
	JZ   vsa_sim_reduce

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y1
	VMOVDQU32 (DI), K7, Y2
	VEXTRACTF128 $0, Y1, X3
	VEXTRACTF128 $0, Y2, X4
	VCVTPS2PD X3, Y5
	VCVTPS2PD X4, Y6
	VFMADD231PD Y6, Y5, K7, Y0

vsa_sim_reduce:
	VHADDPD Y1, Y0, Y0
	VHADDPD Y1, Y1, Y1
	VEXTRACTF128 $0, Y1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+24(FP)
	RET

vsa_sim_zero:
	XORPS X0, X0
	MOVSS X0, ret+24(FP)
	RET
