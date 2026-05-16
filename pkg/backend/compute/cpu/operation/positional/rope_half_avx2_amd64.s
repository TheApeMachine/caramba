#include "textflag.h"

// RoPEHalfAVX2(dst, src, cosTable, sinTable []float64, numPairs int)
// Rotates split-half RoPE pairs: (i, i + numPairs).
TEXT ·RoPEHalfAVX2(SB), NOSPLIT, $0-104
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), SI
	MOVQ cosTable+48(FP), CX
	MOVQ sinTable+72(FP), DX
	MOVQ numPairs+96(FP), BX

	MOVQ BX, R10
	SHLQ $3, R10
	LEAQ (SI)(R10*1), R8
	LEAQ (AX)(R10*1), R9

avx2loop:
	CMPQ BX, $4
	JL scalar_tail

	VMOVUPD (SI), Y0
	VMOVUPD (R8), Y1
	VMOVUPD (CX), Y2
	VMOVUPD (DX), Y3

	VMULPD Y2, Y0, Y4
	VMULPD Y3, Y1, Y5
	VSUBPD Y5, Y4, Y6

	VMULPD Y3, Y0, Y7
	VMULPD Y2, Y1, Y8
	VADDPD Y8, Y7, Y9

	VMOVUPD Y6, (AX)
	VMOVUPD Y9, (R9)

	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, AX
	ADDQ $32, R9
	ADDQ $32, CX
	ADDQ $32, DX
	SUBQ $4, BX
	JMP avx2loop

scalar_tail:
	CMPQ BX, $0
	JLE done

scalar_loop:
	MOVSD (SI), X0
	MOVSD (R8), X1
	MOVSD (CX), X2
	MOVSD (DX), X3

	MOVAPD X0, X4
	MULSD X2, X4
	MOVAPD X1, X5
	MULSD X3, X5
	SUBSD X5, X4
	MOVSD X4, (AX)

	MULSD X3, X0
	MULSD X2, X1
	ADDSD X1, X0
	MOVSD X0, (R9)

	ADDQ $8, SI
	ADDQ $8, R8
	ADDQ $8, AX
	ADDQ $8, R9
	ADDQ $8, CX
	ADDQ $8, DX
	DECQ BX
	JNZ scalar_loop

done:
	VZEROUPPER
	RET
