#include "textflag.h"

DATA losAbsMask<>+0(SB)/4, $0x7fffffff
DATA losAbsMask<>+4(SB)/4, $0x7fffffff
DATA losAbsMask<>+8(SB)/4, $0x7fffffff
DATA losAbsMask<>+12(SB)/4, $0x7fffffff
GLOBL losAbsMask<>(SB), RODATA|NOPTR, $16

// func MseSumFloat32AVX512Asm(predictions, targets *float32, count int) float32
//
// Returns sum((predictions[i]-targets[i])²). Matches f32 scalar accumulation
// order via f64 widen before square and accumulate.
TEXT ·MseSumFloat32AVX512Asm(SB), NOSPLIT, $0-28
	MOVQ predictions+0(FP), SI
	MOVQ targets+8(FP), DI
	MOVQ count+16(FP), CX

	TESTQ CX, CX
	JZ   mse_avx512_zero

	VXORPD Y0, Y0, Y0

mse_avx512_w16:
	CMPQ CX, $16
	JL   mse_avx512_w8

	VMOVUPS Y1, (SI)
	VMOVUPS Y2, (DI)
	VSUBPS  Y2, Y1, Y3
	VEXTRACTF128 $0, Y3, X4
	VCVTPS2PD X4, Y5
	VMULPD  Y5, Y5, Y5
	VADDPD  Y0, Y5, Y0
	VEXTRACTF128 $1, Y3, X4
	VCVTPS2PD X4, Y5
	VMULPD  Y5, Y5, Y5
	VADDPD  Y0, Y5, Y0

	VMOVUPS Y1, 32(SI)
	VMOVUPS Y2, 32(DI)
	VSUBPS  Y2, Y1, Y3
	VEXTRACTF128 $0, Y3, X4
	VCVTPS2PD X4, Y5
	VMULPD  Y5, Y5, Y5
	VADDPD  Y0, Y5, Y0
	VEXTRACTF128 $1, Y3, X4
	VCVTPS2PD X4, Y5
	VMULPD  Y5, Y5, Y5
	VADDPD  Y0, Y5, Y0

	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $16, CX
	JMP  mse_avx512_w16

mse_avx512_w8:
	CMPQ CX, $8
	JL   mse_avx512_w4

	VMOVUPS Y1, (SI)
	VMOVUPS Y2, (DI)
	VSUBPS  Y2, Y1, Y3
	VEXTRACTF128 $0, Y3, X4
	VCVTPS2PD X4, Y5
	VMULPD  Y5, Y5, Y5
	VADDPD  Y0, Y5, Y0
	VEXTRACTF128 $1, Y3, X4
	VCVTPS2PD X4, Y5
	VMULPD  Y5, Y5, Y5
	VADDPD  Y0, Y5, Y0

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  mse_avx512_w8

mse_avx512_w4:
	CMPQ CX, $4
	JL   mse_avx512_w4_tail

	VMOVUPS X1, (SI)
	VMOVUPS X2, (DI)
	VSUBPS  X2, X1, X3
	VCVTPS2PD X3, Y5
	VMULPD  Y5, Y5, Y5
	VADDPD  Y0, Y5, Y0

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  mse_avx512_w4

mse_avx512_w4_tail:
	TESTQ CX, CX
	JZ   mse_avx512_reduce

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y1
	VMOVDQU32 (DI), K7, Y2
	VSUBPS  Y2, Y1, Y3
	VEXTRACTF128 $0, Y3, X4
	VCVTPS2PD X4, Y5
	VMULPD  Y5, Y5, Y5
	VADDPD  Y5, Y0, K7, Y0

mse_avx512_reduce:
	VHADDPD Y1, Y0, Y0
	VHADDPD Y1, Y1, Y1
	VEXTRACTF128 $0, Y1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+24(FP)
	RET

mse_avx512_zero:
	XORPS X0, X0
	MOVSS X0, ret+24(FP)
	RET

// func MaeSumFloat32AVX512Asm(predictions, targets *float32, count int) float32
//
// Returns sum(|predictions[i]-targets[i]|).
TEXT ·MaeSumFloat32AVX512Asm(SB), NOSPLIT, $0-28
	MOVQ predictions+0(FP), SI
	MOVQ targets+8(FP), DI
	MOVQ count+16(FP), CX

	TESTQ CX, CX
	JZ   mae_avx512_zero

	VXORPD Y0, Y0, Y0
	VBROADCASTSS losAbsMask<>(SB), Y7
	VBROADCASTSS losAbsMask<>(SB), X7

mae_avx512_w16:
	CMPQ CX, $16
	JL   mae_avx512_w8

	VMOVUPS Y1, (SI)
	VMOVUPS Y2, (DI)
	VSUBPS  Y2, Y1, Y3
	VANDPS  Y7, Y3, Y3
	VEXTRACTF128 $0, Y3, X4
	VCVTPS2PD X4, Y5
	VADDPD  Y0, Y5, Y0
	VEXTRACTF128 $1, Y3, X4
	VCVTPS2PD X4, Y5
	VADDPD  Y0, Y5, Y0

	VMOVUPS Y1, 32(SI)
	VMOVUPS Y2, 32(DI)
	VSUBPS  Y2, Y1, Y3
	VANDPS  Y7, Y3, Y3
	VEXTRACTF128 $0, Y3, X4
	VCVTPS2PD X4, Y5
	VADDPD  Y0, Y5, Y0
	VEXTRACTF128 $1, Y3, X4
	VCVTPS2PD X4, Y5
	VADDPD  Y0, Y5, Y0

	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $16, CX
	JMP  mae_avx512_w16

mae_avx512_w8:
	CMPQ CX, $8
	JL   mae_avx512_w4

	VMOVUPS Y1, (SI)
	VMOVUPS Y2, (DI)
	VSUBPS  Y2, Y1, Y3
	VANDPS  Y7, Y3, Y3
	VEXTRACTF128 $0, Y3, X4
	VCVTPS2PD X4, Y5
	VADDPD  Y0, Y5, Y0
	VEXTRACTF128 $1, Y3, X4
	VCVTPS2PD X4, Y5
	VADDPD  Y0, Y5, Y0

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  mae_avx512_w8

mae_avx512_w4:
	CMPQ CX, $4
	JL   mae_avx512_w4_tail

	VMOVUPS X1, (SI)
	VMOVUPS X2, (DI)
	VSUBPS  X2, X1, X3
	VANDPS  X7, X3, X3
	VCVTPS2PD X3, Y5
	VADDPD  Y0, Y5, Y0

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  mae_avx512_w4

mae_avx512_w4_tail:
	TESTQ CX, CX
	JZ   mae_avx512_reduce

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y1
	VMOVDQU32 (DI), K7, Y2
	VSUBPS  Y2, Y1, Y3
	VANDPS  Y7, Y3, Y3
	VEXTRACTF128 $0, Y3, X4
	VCVTPS2PD X4, Y5
	VADDPD  Y5, Y0, K7, Y0

mae_avx512_reduce:
	VHADDPD Y1, Y0, Y0
	VHADDPD Y1, Y1, Y1
	VEXTRACTF128 $0, Y1, X0
	CVTSD2SS X0, X0
	MOVSS X0, ret+24(FP)
	RET

mae_avx512_zero:
	XORPS X0, X0
	MOVSS X0, ret+24(FP)
	RET
