#include "textflag.h"

DATA ·argmaxIdx0+0(SB)/8, $0
DATA ·argmaxIdx0+8(SB)/8, $1
DATA ·argmaxIdx0+16(SB)/8, $2
DATA ·argmaxIdx0+24(SB)/8, $3
GLOBL ·argmaxIdx0(SB), RODATA, $32

DATA ·argmaxIdxNext+0(SB)/8, $4
DATA ·argmaxIdxNext+8(SB)/8, $5
DATA ·argmaxIdxNext+16(SB)/8, $6
DATA ·argmaxIdxNext+24(SB)/8, $7
GLOBL ·argmaxIdxNext(SB), RODATA, $32

DATA ·argmaxIdxStep+0(SB)/8, $4
DATA ·argmaxIdxStep+8(SB)/8, $4
DATA ·argmaxIdxStep+16(SB)/8, $4
DATA ·argmaxIdxStep+24(SB)/8, $4
GLOBL ·argmaxIdxStep(SB), RODATA, $32

// argmaxAVX2(xs []float64) int
// Returns the index of the largest element, or 0 for an empty slice.
// NaN values never displace an existing best (matches scalar fallback).
//
// SIMD strategy: 4 doubles per iteration. Y0 holds 4 running best values,
// Y1 holds the corresponding int64 indices. After the vector loop, reduces
// the 4 lanes to a scalar (X9, BX) and finishes any remainder elementwise.
TEXT ·argmaxAVX2(SB), NOSPLIT, $0-32
	MOVQ xs+0(FP), AX
	MOVQ xs_len+8(FP), CX
	XORQ BX, BX
	CMPQ CX, $0
	JLE am_done

	MOVSD (AX), X9                            // running best value
	MOVQ $1, DX                               // scalar default start

	CMPQ CX, $4
	JL am_scalar                              // input too small for SIMD

	// Initialise 4-lane state from xs[0..3].
	VMOVUPD (AX), Y0                          // best values
	VMOVDQU ·argmaxIdx0(SB), Y1               // best indices [0,1,2,3]
	VMOVDQU ·argmaxIdxNext(SB), Y3            // candidate indices [4,5,6,7]
	VMOVDQU ·argmaxIdxStep(SB), Y2            // step [4,4,4,4]
	MOVQ $4, DX

am_vloop:
	MOVQ CX, R8
	SUBQ DX, R8
	CMPQ R8, $4
	JL am_vreduce

	VMOVUPD (AX)(DX*8), Y4
	VCMPPD $0x1E, Y0, Y4, Y5                  // mask = cand > best (GT_OQ; NaN→0)
	VBLENDVPD Y5, Y4, Y0, Y0
	VBLENDVPD Y5, Y3, Y1, Y1
	VPADDQ Y2, Y3, Y3

	ADDQ $4, DX
	JMP am_vloop

am_vreduce:
	// Fold the 4 lanes pairwise: upper 128 vs lower 128.
	VEXTRACTF128 $1, Y0, X10
	VEXTRACTI128 $1, Y1, X11
	VCMPPD $0x1E, X0, X10, X12
	VBLENDVPD X12, X10, X0, X0
	VBLENDVPD X12, X11, X1, X1

	// Fold lane 0 vs lane 1 by swapping the two 64-bit halves and comparing.
	PSHUFD $0x4E, X0, X10
	PSHUFD $0x4E, X1, X11
	UCOMISD X0, X10                           // flags from X10 - X0
	JBE am_lane0
	MOVAPD X10, X0
	MOVAPD X11, X1
am_lane0:
	MOVQ X1, BX                               // extract lane-0 index
	MOVAPD X0, X9                             // best value so far
	VZEROUPPER

am_scalar:
	CMPQ DX, CX
	JGE am_done
	MOVSD (AX)(DX*8), X10
	UCOMISD X9, X10                           // flags from X10 - X9
	JBE am_snext                              // skip on NaN or X10 ≤ X9
	MOVAPD X10, X9
	MOVQ DX, BX
am_snext:
	INCQ DX
	JMP am_scalar

am_done:
	MOVQ BX, ret+24(FP)
	RET
