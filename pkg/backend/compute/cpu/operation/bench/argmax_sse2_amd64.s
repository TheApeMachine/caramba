#include "textflag.h"

DATA ·argmaxIdxSSE20+0(SB)/8, $0
DATA ·argmaxIdxSSE20+8(SB)/8, $0
GLOBL ·argmaxIdxSSE20(SB), RODATA, $16

DATA ·argmaxIdxSSE2Next+0(SB)/8, $1
DATA ·argmaxIdxSSE2Next+8(SB)/8, $2
GLOBL ·argmaxIdxSSE2Next(SB), RODATA, $16

DATA ·argmaxIdxSSE2Step+0(SB)/8, $2
DATA ·argmaxIdxSSE2Step+8(SB)/8, $2
GLOBL ·argmaxIdxSSE2Step(SB), RODATA, $16

// argmaxSSE2(xs []float64) int
// Returns the index of the largest element, or 0 for an empty slice.
// NaN values never displace an existing best.
TEXT ·argmaxSSE2(SB), NOSPLIT, $0-32
	MOVQ xs+0(FP), AX
	MOVQ xs_len+8(FP), CX
	XORQ BX, BX
	CMPQ CX, $0
	JLE am_sse2_done

	MOVSD (AX), X9
	XORQ R8, R8
	UCOMISD X9, X9
	SETPS R8
	TESTQ R8, R8
	JNE am_sse2_done

	MOVQ $1, DX
	CMPQ CX, $2
	JL am_sse2_scalar

	MOVAPD X9, X0
	SHUFPD $0, X0, X0
	MOVOU ·argmaxIdxSSE20(SB), X1
	MOVOU ·argmaxIdxSSE2Next(SB), X3
	MOVOU ·argmaxIdxSSE2Step(SB), X2
	MOVQ $1, DX

am_sse2_vloop:
	MOVQ CX, R8
	SUBQ DX, R8
	CMPQ R8, $2
	JL am_sse2_reduce

	MOVUPD (AX)(DX*8), X4
	MOVAPD X0, X5
	CMPPD X4, X5, $1
	MOVAPD X5, X7

	MOVAPD X5, X6
	ANDPD X4, X6
	ANDNPD X0, X5
	ORPD X6, X5
	MOVAPD X5, X0

	MOVOU X7, X6
	PAND X3, X6
	PANDN X1, X7
	POR X6, X7
	MOVOU X7, X1
	PADDQ X2, X3

	ADDQ $2, DX
	JMP am_sse2_vloop

am_sse2_reduce:
	MOVAPD X0, X10
	PSHUFD $0x4E, X10, X10
	MOVOU X1, X11
	PSHUFD $0x4E, X11, X11
	MOVQ X1, BX
	MOVQ X11, R8
	MOVAPD X0, X9

	UCOMISD X0, X10
	JA am_sse2_lane1
	JE am_sse2_tie
	JMP am_sse2_scalar

am_sse2_tie:
	CMPQ R8, BX
	JGE am_sse2_scalar

am_sse2_lane1:
	MOVQ R8, BX
	MOVAPD X10, X9

am_sse2_scalar:
	CMPQ DX, CX
	JGE am_sse2_done
	MOVSD (AX)(DX*8), X10
	UCOMISD X9, X10
	JBE am_sse2_next
	MOVAPD X10, X9
	MOVQ DX, BX
am_sse2_next:
	INCQ DX
	JMP am_sse2_scalar

am_sse2_done:
	MOVQ BX, ret+24(FP)
	RET
