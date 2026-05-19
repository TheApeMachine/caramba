#include "textflag.h"

// func DropoutFloat32AVX512Asm(
//     dst, src *float32, count int,
//     seedLane *uint32, scale, threshold float32,
// )
//
// Sequential xorshift32 on *seedLane (same as DropoutF32Generic). RNG uses
// XMM vector shifts on lane 0 only; keep/drop uses AVX-512 compare/blend/mul.
TEXT ·DropoutFloat32AVX512Asm(SB), NOSPLIT, $64-40
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX
	MOVQ seedLane+24(FP), R8
	MOVSS scale+32(FP), X15
	MOVSS threshold+36(FP), X14

	VBROADCASTSS X15, Z16
	VBROADCASTSS X14, Z17
	VBROADCASTSS X15, Y16
	VBROADCASTSS X14, Y17

	XORPS X11, X11
	MOVL  (R8), AX
	MOVD  AX, X11

	TESTQ CX, CX
	JZ   drop_done

drop_w16:
	CMPQ CX, $16
	JL   drop_w8

	MOVQ $16, R9
	LEAQ 0(SP), R10
	XORQ R15, R15
	JMP  drop_gen_vec

drop_w8:
	CMPQ CX, $8
	JL   drop_w4

	MOVQ $8, R9
	LEAQ 0(SP), R10
	MOVQ $1, R15
	JMP  drop_gen_vec

drop_w4:
	CMPQ CX, $4
	JL   drop_w4_tail

	MOVQ $4, R9
	LEAQ 0(SP), R10
	MOVQ $2, R15
	JMP  drop_gen_vec

drop_w4_tail:
	TESTQ CX, CX
	JZ   drop_done

	MOVQ CX, R9
	LEAQ 0(SP), R10
	MOVQ $3, R15
	JMP  drop_gen_vec

drop_gen_vec:
	VXORPS Z10, Z10, Z10
	VMOVUPS Z10, 0(SP)
	VMOVUPS Z10, 32(SP)

	VPSLLD $13, X11, X12
	VPXORD X12, X11, X11
	VPSRLD $17, X11, X12
	VPXORD X12, X11, X11
	VPSLLD $5, X11, X12
	VPXORD X12, X11, X11
	VMOVD  X11, (R10)
	ADDQ $4, R10
	DECQ R9
	JNZ  drop_gen_vec

	CMPQ R15, $0
	JE   drop_apply_w16
	CMPQ R15, $1
	JE   drop_apply_w8
	CMPQ R15, $2
	JE   drop_apply_w4
	JMP  drop_apply_tail

drop_apply_w16:
	VXORPS    Z3, Z3, Z3
	VMOVDQU32 (SP), Z2
	VPCMPUD   $1, Z17, Z2, K1
	VMOVUPS   (SI), Z0
	VBLENDMPS Z3, Z0, K1, Z0
	VMULPS    Z16, Z0, Z0
	VMOVUPS   Z0, (DI)

	ADDQ $64, DI
	ADDQ $64, SI
	SUBQ $16, CX
	JMP  drop_w16

drop_apply_w8:
	VXORPS    Y3, Y3, Y3
	VMOVDQU32 (SP), Y2
	VPCMPUD   $1, Y17, Y2, K1
	VMOVUPS   (SI), Y0
	VBLENDMPS Y3, Y0, K1, Y0
	VMULPS    Y16, Y0, Y0
	VMOVUPS   Y0, (DI)

	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $8, CX
	JMP  drop_w16

drop_apply_w4:
	VXORPS   X3, X3, X3
	VMOVDQU  (SP), X2
	VPCMPUD  $1, X14, X2, K1
	VMOVUPS  (SI), X0
	VBLENDMPS X3, X0, K1, X0
	VMULPS   X15, X0, X0
	VMOVUPS  X0, (DI)

	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $4, CX
	JMP  drop_w16

drop_apply_tail:
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VXORPS    X3, X3, X3
	VMOVDQU   (SP), X2
	VPCMPUD   $1, X14, X2, K1
	KANDQ     K7, K1, K1
	VMOVDQU32 (SI), K7, X0
	VBLENDMPS X3, X0, K1, X0
	VMULPS    X15, X0, X0
	VMOVDQU32 X0, K7, (DI)

drop_done:
	VMOVD X11, AX
	MOVL  AX, (R8)
	RET
