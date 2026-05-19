#include "textflag.h"

// func DequantInt8AVX512Asm(dst *float32, src *int8, count int, scale float32, zeroPoint int16)
TEXT ·DequantInt8AVX512Asm(SB), NOSPLIT, $0-30
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX
	MOVSS scale+24(FP), X15
	MOVW zeroPoint+28(FP), R8
	SHLQ $48, R8
	SARQ $48, R8

	VBROADCASTSS X15, Y15
	VPBROADCASTD R8, Y14

	TESTQ CX, CX
	JZ   dequant_i8_done

dequant_i8_w16:
	CMPQ CX, $16
	JL   dequant_i8_w8

	VMOVDQU X0, (SI)
	VPMOVSXBD X0, Y0
	VPSUBD  Y0, Y14, Y0
	VCVTDQ2PS Y0, Y0
	VMULPS  Y0, Y15, Y0
	VMOVUPS Y0, (DI)

	VMOVDQU X0, 8(SI)
	VPMOVSXBD X0, Y0
	VPSUBD  Y0, Y14, Y0
	VCVTDQ2PS Y0, Y0
	VMULPS  Y0, Y15, Y0
	VMOVUPS Y0, 32(DI)

	ADDQ $16, SI
	ADDQ $64, DI
	SUBQ $16, CX
	JMP  dequant_i8_w16

dequant_i8_w8:
	CMPQ CX, $8
	JL   dequant_i8_w4_tail

	VMOVDQU X0, (SI)
	VPMOVSXBD X0, Y0
	VPSUBD  Y0, Y14, Y0
	VCVTDQ2PS Y0, Y0
	VMULPS  Y0, Y15, Y0
	VMOVUPS Y0, (DI)

	ADDQ $8, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  dequant_i8_w16

dequant_i8_w4_tail:
	TESTQ CX, CX
	JZ   dequant_i8_done

dequant_i8_scalar_loop:
	MOVB (SI), R9
	SHLQ $56, R9
	SARQ $56, R9
	SUBQ R8, R9
	IMULQ $4, R9
	CVTSQ2SS R9, X0
	MULSS X15, X0
	MOVSS X0, (DI)

	ADDQ $1, SI
	ADDQ $4, DI
	DECQ CX
	JNZ  dequant_i8_scalar_loop

dequant_i8_done:
	RET
