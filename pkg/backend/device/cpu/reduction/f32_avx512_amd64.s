#include "textflag.h"

DATA redAbsMask<>+0(SB)/4, $0x7fffffff
DATA redAbsMask<>+4(SB)/4, $0x7fffffff
DATA redAbsMask<>+8(SB)/4, $0x7fffffff
DATA redAbsMask<>+12(SB)/4, $0x7fffffff
DATA redAbsMask<>+16(SB)/4, $0x7fffffff
DATA redAbsMask<>+20(SB)/4, $0x7fffffff
DATA redAbsMask<>+24(SB)/4, $0x7fffffff
DATA redAbsMask<>+28(SB)/4, $0x7fffffff
DATA redAbsMask<>+32(SB)/4, $0x7fffffff
DATA redAbsMask<>+36(SB)/4, $0x7fffffff
DATA redAbsMask<>+40(SB)/4, $0x7fffffff
DATA redAbsMask<>+44(SB)/4, $0x7fffffff
DATA redAbsMask<>+48(SB)/4, $0x7fffffff
DATA redAbsMask<>+52(SB)/4, $0x7fffffff
DATA redAbsMask<>+56(SB)/4, $0x7fffffff
DATA redAbsMask<>+60(SB)/4, $0x7fffffff
GLOBL redAbsMask<>(SB), RODATA|NOPTR, $64

DATA redOneF32<>+0(SB)/4, $0x3f800000
GLOBL redOneF32<>(SB), RODATA|NOPTR, $4

// func SumFloat32AVX512Asm(src *float32, count int) float32
TEXT ·SumFloat32AVX512Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   sum_avx512_zero

	VXORPD Z0, Z0, Z0

sum_avx512_w16:
	CMPQ CX, $16
	JL   sum_avx512_w8

	VMOVUPS (SI), Z2
	VEXTRACTF128 $0, Z2, Y4
	VCVTPS2PD Z3, Y4
	VADDPD  Z0, Z3, Z0
	VEXTRACTF128 $1, Z2, Y4
	VCVTPS2PD Z3, Y4
	VADDPD  Z0, Z3, Z0
	VEXTRACTF128 $2, Z2, Y4
	VCVTPS2PD Z3, Y4
	VADDPD  Z0, Z3, Z0
	VEXTRACTF128 $3, Z2, Y4
	VCVTPS2PD Z3, Y4
	VADDPD  Z0, Z3, Z0

	ADDQ $64, SI
	SUBQ $16, CX
	JMP  sum_avx512_w16

sum_avx512_w8:
	CMPQ CX, $8
	JL   sum_avx512_w4

	VMOVUPS (SI), Y2
	VCVTPS2PD Z3, Y2
	VADDPD  Z0, Z3, Z0
	VEXTRACTF128 $1, Y2, X4
	VCVTPS2PD Z5, X4
	VADDPD  Z0, Z5, Z0

	ADDQ $32, SI
	SUBQ $8, CX
	JMP  sum_avx512_w8

sum_avx512_w4:
	CMPQ CX, $4
	JL   sum_avx512_w4_tail

	VMOVUPS (SI), X2
	VCVTPS2PD Z3, X2
	VADDPD  Z0, Z3, Z0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  sum_avx512_w4

sum_avx512_w4_tail:
	TESTQ CX, CX
	JZ   sum_avx512_reduce

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y2
	VCVTPS2PD Z3, Y2
	VADDPD  Z3, Z0, K7, Z0

sum_avx512_reduce:
	VEXTRACTF128 $0, Z0, X0
	VEXTRACTF128 $1, Z0, X1
	VEXTRACTF128 $2, Z0, X2
	VEXTRACTF128 $3, Z0, X3
	VADDPD  X1, X0, X0
	VADDPD  X2, X0, X0
	VADDPD  X3, X0, X0
	VHADDPD X0, X0, X0
	VHADDPD X0, X0, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+16(FP)
	RET

sum_avx512_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func ProdFloat32AVX512Asm(src *float32, count int) float32
TEXT ·ProdFloat32AVX512Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   prod_avx512_zero

	VBROADCASTSS redOneF32<>(SB), Z0

prod_avx512_w16:
	CMPQ CX, $16
	JL   prod_avx512_w8

	VMOVUPS (SI), Z1
	VMULPS  Z1, Z0, Z0

	ADDQ $64, SI
	SUBQ $16, CX
	JMP  prod_avx512_w16

prod_avx512_w8:
	CMPQ CX, $8
	JL   prod_avx512_w4

	VMOVUPS (SI), Y1
	VMULPS  Y1, Z0, Z0

	ADDQ $32, SI
	SUBQ $8, CX
	JMP  prod_avx512_w8

prod_avx512_w4:
	CMPQ CX, $4
	JL   prod_avx512_w4_tail

	VMOVUPS (SI), X1
	VMULPS  X1, Z0, Z0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  prod_avx512_w4

prod_avx512_w4_tail:
	TESTQ CX, CX
	JZ   prod_avx512_fold

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y1
	VMULPS  Y1, Z0, K7, Z0

prod_avx512_fold:
	VEXTRACTF128 $0, Z0, Y1
	VEXTRACTF128 $1, Z0, Y2
	VEXTRACTF128 $2, Z0, Y3
	VEXTRACTF128 $3, Z0, Y4
	VMULPS  Y2, Y1, Y1
	VMULPS  Y3, Y1, Y1
	VMULPS  Y4, Y1, Y1
	VHADDPS Y1, Y1, Y1
	VHADDPS Y1, Y1, Y1
	VEXTRACTF128 $0, Y1, X0
	MOVSS X0, ret+16(FP)
	RET

prod_avx512_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func ReduceMaxFloat32AVX512Asm(src *float32, count int) float32
TEXT ·ReduceMaxFloat32AVX512Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   max_avx512_zero

	MOVSS (SI), X0
	VBROADCASTSS X0, Z0
	ADDQ $4, SI
	DECQ CX

max_avx512_w16:
	CMPQ CX, $16
	JL   max_avx512_w8

	VMOVUPS (SI), Z1
	VMAXPS  Z1, Z0, Z0

	ADDQ $64, SI
	SUBQ $16, CX
	JMP  max_avx512_w16

max_avx512_w8:
	CMPQ CX, $8
	JL   max_avx512_w4

	VMOVUPS (SI), Y1
	VMAXPS  Y1, Z0, Z0

	ADDQ $32, SI
	SUBQ $8, CX
	JMP  max_avx512_w8

max_avx512_w4:
	CMPQ CX, $4
	JL   max_avx512_w4_tail

	VMOVUPS (SI), Y1
	VMAXPS  Y1, Z0, Z0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  max_avx512_w4

max_avx512_w4_tail:
	TESTQ CX, CX
	JZ   max_avx512_extract

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y1
	VMAXPS  Y1, Z0, K7, Z0

max_avx512_extract:
	VEXTRACTF128 $0, Z0, Y1
	VEXTRACTF128 $1, Z0, Y2
	VEXTRACTF128 $2, Z0, Y3
	VEXTRACTF128 $3, Z0, Y4
	VMAXPS  Y2, Y1, Y1
	VMAXPS  Y3, Y1, Y1
	VMAXPS  Y4, Y1, Y1
	VHADDPS Y1, Y1, Y1
	VHADDPS Y1, Y1, Y1
	VEXTRACTF128 $0, Y1, X0
	MOVSS X0, ret+16(FP)
	RET

max_avx512_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func ReduceMinFloat32AVX512Asm(src *float32, count int) float32
TEXT ·ReduceMinFloat32AVX512Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   min_avx512_zero

	MOVSS (SI), X0
	VBROADCASTSS X0, Z0
	ADDQ $4, SI
	DECQ CX

min_avx512_w16:
	CMPQ CX, $16
	JL   min_avx512_w8

	VMOVUPS (SI), Z1
	VMINPS  Z1, Z0, Z0

	ADDQ $64, SI
	SUBQ $16, CX
	JMP  min_avx512_w16

min_avx512_w8:
	CMPQ CX, $8
	JL   min_avx512_w4

	VMOVUPS (SI), Y1
	VMINPS  Y1, Z0, Z0

	ADDQ $32, SI
	SUBQ $8, CX
	JMP  min_avx512_w8

min_avx512_w4:
	CMPQ CX, $4
	JL   min_avx512_w4_tail

	VMOVUPS (SI), Y1
	VMINPS  Y1, Z0, Z0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  min_avx512_w4

min_avx512_w4_tail:
	TESTQ CX, CX
	JZ   min_avx512_extract

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y1
	VMINPS  Y1, Z0, K7, Z0

min_avx512_extract:
	VEXTRACTF128 $0, Z0, Y1
	VEXTRACTF128 $1, Z0, Y2
	VEXTRACTF128 $2, Z0, Y3
	VEXTRACTF128 $3, Z0, Y4
	VMINPS  Y2, Y1, Y1
	VMINPS  Y3, Y1, Y1
	VMINPS  Y4, Y1, Y1
	VHADDPS Y1, Y1, Y1
	VHADDPS Y1, Y1, Y1
	VEXTRACTF128 $0, Y1, X0
	MOVSS X0, ret+16(FP)
	RET

min_avx512_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET

// func L1NormFloat32AVX512Asm(src *float32, count int) float32
TEXT ·L1NormFloat32AVX512Asm(SB), NOSPLIT, $0-20
	MOVQ src+0(FP), SI
	MOVQ count+8(FP), CX
	TESTQ CX, CX
	JZ   l1_avx512_zero

	VXORPD Z0, Z0, Z0
	VMOVUPS redAbsMask<>(SB), Z6

l1_avx512_w16:
	CMPQ CX, $16
	JL   l1_avx512_w8

	VMOVUPS (SI), Z2
	VANDPS  Z6, Z2, Z2
	VEXTRACTF128 $0, Z2, Y4
	VCVTPS2PD Z3, Y4
	VADDPD  Z0, Z3, Z0
	VEXTRACTF128 $1, Z2, Y4
	VCVTPS2PD Z3, Y4
	VADDPD  Z0, Z3, Z0
	VEXTRACTF128 $2, Z2, Y4
	VCVTPS2PD Z3, Y4
	VADDPD  Z0, Z3, Z0
	VEXTRACTF128 $3, Z2, Y4
	VCVTPS2PD Z3, Y4
	VADDPD  Z0, Z3, Z0

	ADDQ $64, SI
	SUBQ $16, CX
	JMP  l1_avx512_w16

l1_avx512_w8:
	CMPQ CX, $8
	JL   l1_avx512_w4

	VMOVUPS (SI), Y2
	VEXTRACTF128 $0, Z6, Y7
	VANDPS  Y7, Y2, Y2
	VCVTPS2PD Z3, Y2
	VADDPD  Z0, Z3, Z0
	VEXTRACTF128 $1, Y2, X4
	VCVTPS2PD Z5, X4
	VADDPD  Z0, Z5, Z0

	ADDQ $32, SI
	SUBQ $8, CX
	JMP  l1_avx512_w8

l1_avx512_w4:
	CMPQ CX, $4
	JL   l1_avx512_w4_tail

	VMOVUPS (SI), Y2
	VEXTRACTF128 $0, Z6, Y7
	VANDPS  Y7, Y2, Y2
	VCVTPS2PD Z3, Y2
	VADDPD  Z0, Z3, Z0

	ADDQ $16, SI
	SUBQ $4, CX
	JMP  l1_avx512_w4

l1_avx512_w4_tail:
	TESTQ CX, CX
	JZ   l1_avx512_reduce

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y2
	VEXTRACTF128 $0, Z6, Y7
	VANDPS  Y7, Y2, Y2
	VCVTPS2PD Z3, Y2
	VADDPD  Z3, Z0, K7, Z0

l1_avx512_reduce:
	VEXTRACTF128 $0, Z0, X0
	VEXTRACTF128 $1, Z0, X1
	VEXTRACTF128 $2, Z0, X2
	VEXTRACTF128 $3, Z0, X3
	VADDPD  X1, X0, X0
	VADDPD  X2, X0, X0
	VADDPD  X3, X0, X0
	VHADDPD X0, X0, X0
	VHADDPD X0, X0, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+16(FP)
	RET

l1_avx512_zero:
	XORPS X0, X0
	MOVSS X0, ret+16(FP)
	RET
