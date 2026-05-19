// SPDX-License-Identifier: Apache-2.0
// AVX-512 float32 masking kernels: apply-mask add, causal mask, ALiBi bias.
#include "textflag.h"

DATA maskZero<>+0(SB)/4, $0.0
DATA maskZero<>+4(SB)/4, $0.0
DATA maskZero<>+8(SB)/4, $0.0
DATA maskZero<>+12(SB)/4, $0.0
GLOBL maskZero<>(SB), RODATA|NOPTR, $16

DATA maskNegInf<>+0(SB)/4, $-0x1.0p127
DATA maskNegInf<>+4(SB)/4, $-0x1.0p127
DATA maskNegInf<>+8(SB)/4, $-0x1.0p127
DATA maskNegInf<>+12(SB)/4, $-0x1.0p127
GLOBL maskNegInf<>(SB), RODATA|NOPTR, $16

DATA maskIota16<>+0(SB)/4, $0
DATA maskIota16<>+4(SB)/4, $1
DATA maskIota16<>+8(SB)/4, $2
DATA maskIota16<>+12(SB)/4, $3
DATA maskIota16<>+16(SB)/4, $4
DATA maskIota16<>+20(SB)/4, $5
DATA maskIota16<>+24(SB)/4, $6
DATA maskIota16<>+28(SB)/4, $7
DATA maskIota16<>+32(SB)/4, $8
DATA maskIota16<>+36(SB)/4, $9
DATA maskIota16<>+40(SB)/4, $10
DATA maskIota16<>+44(SB)/4, $11
DATA maskIota16<>+48(SB)/4, $12
DATA maskIota16<>+52(SB)/4, $13
DATA maskIota16<>+56(SB)/4, $14
DATA maskIota16<>+60(SB)/4, $15
GLOBL maskIota16<>(SB), RODATA|NOPTR, $64

// func ApplyMaskFloat32AVX512Asm(input, mask, output *float32, count int)
TEXT ·ApplyMaskFloat32AVX512Asm(SB), NOSPLIT, $0-32
	MOVQ input+0(FP), DI
	MOVQ mask+8(FP), SI
	MOVQ output+16(FP), R8
	MOVQ count+24(FP), CX

mask_apply_w16:
	CMPQ CX, $16
	JL   mask_apply_w8

	VMOVUPS Y0, (DI)
	VMOVUPS Y1, (SI)
	VADDPS  Y1, Y0, Y0
	VMOVUPS Y0, (R8)
	VMOVUPS Y2, 32(DI)
	VMOVUPS Y3, 32(SI)
	VADDPS  Y3, Y2, Y2
	VMOVUPS Y2, 32(R8)

	ADDQ $64, DI
	ADDQ $64, SI
	ADDQ $64, R8
	SUBQ $16, CX
	JMP  mask_apply_w16

mask_apply_w8:
	CMPQ CX, $8
	JL   mask_apply_w4

	VMOVUPS Y0, (DI)
	VMOVUPS Y1, (SI)
	VADDPS  Y1, Y0, Y0
	VMOVUPS Y0, (R8)

	ADDQ $32, DI
	ADDQ $32, SI
	ADDQ $32, R8
	SUBQ $8, CX
	JMP  mask_apply_w8

mask_apply_w4:
	CMPQ CX, $4
	JL   mask_apply_w4_tail

	VMOVUPS X0, (DI)
	VMOVUPS X1, (SI)
	VADDPS  X1, X0, X0
	VMOVUPS X0, (R8)

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R8
	SUBQ $4, CX
	JMP  mask_apply_w4

mask_apply_w4_tail:
	TESTQ CX, CX
	JZ   mask_apply_done

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (DI), K7, Y0
	VMOVDQU32 (SI), K7, Y1
	VADDPS  Y1, Y0, Y0
	VMOVDQU32 Y0, K7, (R8)

mask_apply_done:
	RET

// func CausalMaskFloat32AVX512Asm(output *float32, seqQ, seqK int)
TEXT ·CausalMaskFloat32AVX512Asm(SB), NOSPLIT, $0-24
	MOVQ output+0(FP), DI
	MOVQ seqQ+8(FP), R10
	MOVQ seqK+16(FP), BX

	VBROADCASTSS maskZero<>(SB), Z0
	VBROADCASTSS maskNegInf<>(SB), Z1

	XORQ R11, R11

causal_row:
	CMPQ R11, R10
	JGE  causal_done

	MOVQ R11, AX
	INCQ AX
	CMPQ AX, BX
	JLE  causal_zero_len_ok
	MOVQ BX, AX

causal_zero_len_ok:
	MOVQ AX, CX

causal_zero_w16:
	CMPQ CX, $16
	JL   causal_zero_w8

	VMOVUPS Z0, (DI)
	VMOVUPS Z0, 64(DI)

	ADDQ $128, DI
	SUBQ $16, CX
	JMP  causal_zero_w16

causal_zero_w8:
	CMPQ CX, $8
	JL   causal_zero_w4

	VMOVUPS Y0, (DI)

	ADDQ $32, DI
	SUBQ $8, CX
	JMP  causal_zero_w8

causal_zero_w4:
	CMPQ CX, $4
	JL   causal_zero_w4_tail

	VMOVUPS X0, (DI)

	ADDQ $16, DI
	SUBQ $4, CX
	JMP  causal_zero_w4

causal_zero_w4_tail:
	TESTQ CX, CX
	JZ   causal_zero_done

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 Z0, K7, (DI)

causal_zero_done:
	MOVQ seqK+16(FP), BX
	MOVQ R11, AX
	INCQ AX
	CMPQ AX, BX
	JGE  causal_next_row

	MOVQ BX, R12
	SUBQ AX, R12
	MOVQ R12, CX

causal_inf_w16:
	CMPQ CX, $16
	JL   causal_inf_w8

	VMOVUPS Z1, (DI)
	VMOVUPS Z1, 64(DI)

	ADDQ $128, DI
	SUBQ $16, CX
	JMP  causal_inf_w16

causal_inf_w8:
	CMPQ CX, $8
	JL   causal_inf_w4

	VMOVUPS Y1, (DI)

	ADDQ $32, DI
	SUBQ $8, CX
	JMP  causal_inf_w8

causal_inf_w4:
	CMPQ CX, $4
	JL   causal_inf_w4_tail

	VMOVUPS X1, (DI)

	ADDQ $16, DI
	SUBQ $4, CX
	JMP  causal_inf_w4

causal_inf_w4_tail:
	TESTQ CX, CX
	JZ   causal_next_row

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 Z1, K7, (DI)

causal_next_row:
	INCQ R11
	JMP  causal_row

causal_done:
	RET

// func ALiBiBiasFloat32AVX512Asm(scores, slope, output *float32, seqQ, seqK int)
TEXT ·ALiBiBiasFloat32AVX512Asm(SB), NOSPLIT, $0-40
	MOVQ scores+0(FP), SI
	MOVQ slope+8(FP), R9
	MOVQ output+16(FP), DI
	MOVQ seqQ+24(FP), R10
	MOVQ seqK+32(FP), BX

	VMOVDQA32 maskIota16<>(SB), Z14

	XORQ R11, R11

alibi_row:
	CMPQ R11, R10
	JGE  alibi_done

	VBROADCASTSS (R9), Z15
	VBROADCASTSS (R9), Y15
	VBROADCASTSS (R9), X15
	VMOVD R11, X13
	VPBROADCASTD X13, Z13
	VPBROADCASTD X13, Y13

	XORQ R12, R12

alibi_col:
	MOVQ BX, CX
	SUBQ R12, CX
	JZ   alibi_row_done

	CMPQ CX, $16
	JL   alibi_col_w8

	VMOVUPS Z0, (SI)
	VMOVD R12, X10
	VPBROADCASTD X10, Z12
	VPADDD  Z14, Z12, Z12
	VPSUBD  Z12, Z13, Z11
	VCVTDQ2PS Z11, Z10
	VXORPS  Z9, Z9, Z9
	VCMPPS  $1, Z10, Z9, K1
	VMULPS  Z15, Z10, Z16
	VSUBPS  Z16, Z0, Z1
	VBLENDMPS Z0, Z1, K1, Z0
	VMOVUPS Z0, (DI)

	ADDQ $64, SI
	ADDQ $64, DI
	ADDQ $16, R12
	JMP  alibi_col

alibi_col_w8:
	CMPQ CX, $8
	JL   alibi_col_w4

	VMOVUPS Y0, (SI)
	VMOVD R12, X10
	VPBROADCASTD X10, Y12
	VPADDD  maskIota16<>(SB), Y12, Y12
	VPSUBD  Y12, Y13, Y11
	VCVTDQ2PS Y11, Y10
	VXORPS  Y9, Y9, Y9
	VCMPPS  $1, Y10, Y9, K1
	VMULPS  Y15, Y10, Y16
	VSUBPS  Y16, Y0, Y1
	VBLENDMPS Y0, Y1, K1, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $8, R12
	JMP  alibi_col

alibi_col_w4:
	CMPQ CX, $4
	JL   alibi_col_w4_tail

	VMOVUPS X0, (SI)
	VMOVD R12, X10
	VPBROADCASTD X10, X11
	VPADDD  maskIota16<>(SB), X11, X11
	VPSUBD  X11, X13, X12
	VCVTDQ2PS X12, X10
	VXORPS  X9, X9, X9
	VCMPPS  $1, X10, X9, K1
	VMULPS  X15, X10, X11
	VSUBPS  X11, X0, X1
	VBLENDMPS X0, X1, K1, X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	ADDQ $4, R12
	JMP  alibi_col

alibi_col_w4_tail:
	TESTQ CX, CX
	JZ   alibi_row_done

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, X0
	VMOVD R12, X10
	VPBROADCASTD X10, X11
	VPADDD  maskIota16<>(SB), X11, X11
	VPSUBD  X11, X13, X12
	VCVTDQ2PS X12, X10
	VXORPS  X9, X9, X9
	VCMPPS  $1, X10, X9, K2
	KANDW K2, K7, K1
	VMULPS  X15, X10, X11
	VSUBPS  X11, X0, X1
	VBLENDMPS X0, X1, K1, X0
	VMOVDQU32 X0, K7, (DI)

alibi_row_done:
	INCQ R11
	JMP  alibi_row

alibi_done:
	RET
