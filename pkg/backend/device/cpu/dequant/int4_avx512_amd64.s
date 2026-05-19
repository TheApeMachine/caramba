#include "textflag.h"

DATA dequantI4Mask0F<>+0(SB)/8, $0x0F0F0F0F0F0F0F0F
GLOBL dequantI4Mask0F<>(SB), RODATA, $8

// func DequantInt4AVX512Asm(dst *float32, src *byte, count int, scale float32, zeroPoint int8)
TEXT ·DequantInt4AVX512Asm(SB), NOSPLIT, $0-29
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX
	MOVSS scale+24(FP), X15
	MOVB zeroPoint+28(FP), R8
	SHLQ $56, R8
	SARQ $56, R8

	VBROADCASTSS X15, Y15
	VPBROADCASTD R8, Y14
	VMOVDQU X7, dequantI4Mask0F<>(SB)

	TESTQ CX, CX
	JZ   dequant_i4_done

dequant_i4_w16:
	CMPQ CX, $16
	JL   dequant_i4_w8

	VMOVDQU X0, (SI)

	VPMOVZXBW X0, Y0
	VPAND X1, X0, X7
	VPSLLW $12, X1, X1
	VPSRAW $12, X1, X1
	VPSRLW $4, Y0, Y2
	VPSLLW $12, Y2, Y2
	VPSRAW $12, Y2, Y2

	VPUNPCKLWD X1, X2, X3
	VPUNPCKHWD X1, X2, X4

	VPMOVSXBD X3, Y5
	VPMOVSXBD X4, Y6

	VPSUBD Y5, Y14, Y5
	VPSUBD Y6, Y14, Y6

	VCVTDQ2PS Y5, Y5
	VCVTDQ2PS Y6, Y6

	VMULPS Y5, Y15, Y5
	VMULPS Y6, Y15, Y6

	VMOVUPS Y5, (DI)
	VMOVUPS Y6, 32(DI)

	ADDQ $8, SI
	ADDQ $64, DI
	SUBQ $16, CX
	JMP  dequant_i4_w16

dequant_i4_w8:
	CMPQ CX, $8
	JL   dequant_i4_scalar_tail

	VMOVDQU X0, (SI)

	VPMOVZXBW X0, Y0
	VPAND X1, X0, X7
	VPSLLW $12, X1, X1
	VPSRAW $12, X1, X1
	VPSRLW $4, Y0, Y2
	VPSLLW $12, Y2, Y2
	VPSRAW $12, Y2, Y2

	VPUNPCKLWD X1, X2, X3

	VPMOVSXBD X3, Y5

	VPSUBD Y5, Y14, Y5
	VCVTDQ2PS Y5, Y5
	VMULPS Y5, Y15, Y5
	VMOVUPS Y5, (DI)

	ADDQ $4, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  dequant_i4_w16

dequant_i4_scalar_tail:
	TESTQ CX, CX
	JZ   dequant_i4_done

	MOVQ $0, R10

dequant_i4_scalar_loop:
	MOVB (SI), R9

	CMPQ R10, $0
	JEQ  dequant_i4_take_lo
	SHRQ $4, R9

dequant_i4_take_lo:
	ANDQ $15, R9
	SHLQ $60, R9
	SARQ $60, R9
	SUBQ R8, R9
	IMULQ $4, R9
	CVTSQ2SS R9, X0
	MULSS X15, X0
	MOVSS X0, (DI)

	ADDQ $4, DI
	XORQ $1, R10
	CMPQ R10, $1
	JNE  dequant_i4_next_iter
	ADDQ $1, SI

dequant_i4_next_iter:
	DECQ CX
	JNZ  dequant_i4_scalar_loop

dequant_i4_done:
	RET
