#include "textflag.h"

// func CopyRowFloat32AVX512Asm(dst, src *float32, hidden int)
TEXT ·CopyRowFloat32AVX512Asm(SB), NOSPLIT, $0-20
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ hidden+16(FP), CX

emb_copy_w16:
	CMPQ CX, $16
	JL   emb_copy_w8

	VMOVUPS Y0, (SI)
	VMOVUPS Y0, (DI)
	VMOVUPS Y1, 32(SI)
	VMOVUPS Y1, 32(DI)

	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $16, CX
	JMP  emb_copy_w16

emb_copy_w8:
	CMPQ CX, $8
	JL   emb_copy_w4

	VMOVUPS Y0, (SI)
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  emb_copy_w16

emb_copy_w4:
	CMPQ CX, $4
	JL   emb_copy_w4_tail

	VMOVUPS X0, (SI)
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  emb_copy_w4

emb_copy_w4_tail:
	TESTQ CX, CX
	JZ   emb_copy_done

	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y0
	VMOVDQU32 Y0, K7, (DI)

emb_copy_done:
	RET

// func AddRowFloat32AVX512Asm(dst, src *float32, hidden int)
TEXT ·AddRowFloat32AVX512Asm(SB), NOSPLIT, $0-20
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ hidden+16(FP), CX

emb_add_w16:
	CMPQ CX, $16
	JL   emb_add_w8

	VMOVUPS Y0, (SI)
	VMOVUPS Y1, (DI)
	VADDPS  Y0, Y1, Y0
	VMOVUPS Y0, (DI)
	VMOVUPS Y2, 32(SI)
	VMOVUPS Y3, 32(DI)
	VADDPS  Y2, Y3, Y2
	VMOVUPS Y2, 32(DI)

	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $16, CX
	JMP  emb_add_w16

emb_add_w8:
	CMPQ CX, $8
	JL   emb_add_w4

	VMOVUPS Y0, (SI)
	VMOVUPS Y1, (DI)
	VADDPS  Y0, Y1, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  emb_add_w16

emb_add_w4:
	CMPQ CX, $4
	JL   emb_add_w4_tail

	VMOVUPS X0, (SI)
	VMOVUPS X1, (DI)
	VADDPS  X0, X1, X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  emb_add_w4

emb_add_w4_tail:
	TESTQ CX, CX
	JZ   emb_add_done

	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y0
	VMOVDQU32 (DI), K7, Y1
	VADDPS  Y0, Y1, Y0
	VMOVDQU32 Y0, K7, (DI)

emb_add_done:
	RET
