#include "textflag.h"

// func DropoutFloat32AVX512Asm(
//     dst, src *float32, count int,
//     seedLane *uint32, scale, threshold float32,
// )
//
// count must be a multiple of 16. Advances *seedLane with sequential
// xorshift32 (same as dropoutFloat32ScalarLane / DropoutF32Generic).
// keep mask: uint32(rand) < uint32(threshold bits); output = src * scale.
TEXT ·DropoutFloat32AVX512Asm(SB), NOSPLIT, $64-40
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX
	MOVQ seedLane+24(FP), R8
	MOVSS scale+32(FP), X15
	MOVSS threshold+36(FP), X14

	VBROADCASTSS X15, Z16
	VBROADCASTSS X14, Z17

	MOVL (R8), R11

do_w16:
	CMPQ CX, $16
	JL   do_done

	MOVQ $16, R9
	LEAQ 0(SP), R10

do_gen_rand:
	MOVL R11, AX
	SHLL $13, AX
	XORL AX, R11
	MOVL R11, AX
	SHRL $17, AX
	XORL AX, R11
	MOVL R11, AX
	SHLL $5, AX
	XORL AX, R11
	MOVL R11, (R10)

	ADDQ $4, R10
	DECQ R9
	JNZ  do_gen_rand

	VXORPS   Z3, Z3, Z3
	VMOVDQU32 (SP), Z2
	VPCMPUD  $1, Z17, Z2, K1
	VMOVUPS  (SI), Z0
	VBLENDMPS Z3, Z0, K1, Z0
	VMULPS   Z16, Z0, Z0
	VMOVUPS  Z0, (DI)

	ADDQ $64, DI
	ADDQ $64, SI
	SUBQ $16, CX
	JMP  do_w16

do_done:
	MOVL R11, (R8)
	RET
