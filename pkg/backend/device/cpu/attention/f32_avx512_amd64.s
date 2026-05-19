#include "textflag.h"

// func FlashAttentionOnlineUpdateAVX512Asm(
//     acc, valueRow *float32,
//     alpha, shifted float32,
//     n int,
// )
//
// acc[i] = acc[i]*alpha + valueRow[i]*shifted for i in [0,n).
TEXT ·FlashAttentionOnlineUpdateAVX512Asm(SB), NOSPLIT, $0-32
	MOVD acc+0(FP), SI
	MOVD valueRow+8(FP), DI
	MOVQ n+24(FP), CX

	VBROADCASTSS alpha+16(FP), Y16
	VBROADCASTSS shifted+20(FP), Y17

flash_upd_avx512_w16:
	CMPQ CX, $16
	JL   flash_upd_avx512_w8

	VMOVUPS (SI), Y0
	VMOVUPS (DI), Y2
	VMULPS  Y16, Y0, Y0
	VMULPS  Y17, Y2, Y2
	VADDPS  Y2, Y0, Y0
	VMOVUPS Y0, (SI)

	VMOVUPS 32(SI), Y0
	VMOVUPS 32(DI), Y2
	VMULPS  Y16, Y0, Y0
	VMULPS  Y17, Y2, Y2
	VADDPS  Y2, Y0, Y0
	VMOVUPS Y0, 32(SI)

	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $16, CX
	JMP  flash_upd_avx512_w16

flash_upd_avx512_w8:
	CMPQ CX, $8
	JL   flash_upd_avx512_w4

	VMOVUPS (SI), Y0
	VMOVUPS (DI), Y2
	VMULPS  Y16, Y0, Y0
	VMULPS  Y17, Y2, Y2
	VADDPS  Y2, Y0, Y0
	VMOVUPS Y0, (SI)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  flash_upd_avx512_w8

flash_upd_avx512_w4:
	CMPQ CX, $4
	JL   flash_upd_avx512_w4_tail

	VMOVUPS (SI), X0
	VMOVUPS (DI), X2
	VMULPS  X16, X0, X0
	VMULPS  X17, X2, X2
	VADDPS  X2, X0, X0
	VMOVUPS X0, (SI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  flash_upd_avx512_w4

flash_upd_avx512_w4_tail:
	TESTQ CX, CX
	JZ   flash_upd_avx512_done

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7
	VMOVDQU32 (SI), K7, Y0
	VMOVDQU32 (DI), K7, Y2
	VMULPS  Y16, Y0, K7, Y0
	VMULPS  Y17, Y2, K7, Y2
	VADDPS  Y2, Y0, K7, Y0
	VMOVDQU32 Y0, K7, (SI)

flash_upd_avx512_done:
	RET

// func FlashAttentionScaleAVX512Asm(
//     out, acc *float32,
//     invNormalizer float32,
//     n int,
// )
//
// out[i] = acc[i] * invNormalizer for i in [0,n).
TEXT ·FlashAttentionScaleAVX512Asm(SB), NOSPLIT, $0-32
	MOVD out+0(FP), DI
	MOVD acc+8(FP), SI
	MOVQ n+24(FP), CX

	VBROADCASTSS invNormalizer+16(FP), Y16

flash_scale_avx512_w16:
	CMPQ CX, $16
	JL   flash_scale_avx512_w8

	VMOVUPS (SI), Y0
	VMULPS  Y0, Y16, Y0
	VMOVUPS Y0, (DI)

	VMOVUPS 32(SI), Y0
	VMULPS  Y0, Y16, Y0
	VMOVUPS Y0, 32(DI)

	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $16, CX
	JMP  flash_scale_avx512_w16

flash_scale_avx512_w8:
	CMPQ CX, $8
	JL   flash_scale_avx512_w4

	VMOVUPS (SI), Y0
	VMULPS  Y0, Y16, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  flash_scale_avx512_w8

flash_scale_avx512_w4:
	CMPQ CX, $4
	JL   flash_scale_avx512_w4_tail

	VMOVUPS (SI), X0
	VMULPS  X0, X16, X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  flash_scale_avx512_w4

flash_scale_avx512_w4_tail:
	TESTQ CX, CX
	JZ   flash_scale_avx512_done

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7
	VMOVDQU32 (SI), K7, Y0
	VMULPS  Y0, Y16, K7, Y0
	VMOVDQU32 Y0, K7, (DI)

flash_scale_avx512_done:
	RET
