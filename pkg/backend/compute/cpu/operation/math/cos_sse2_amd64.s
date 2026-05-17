#include "textflag.h"

// cosVecSSE2(dst, src []float64) — 2 lanes per iteration. Same
// reduction as sinVecSSE2; differs in octant→polynomial selection
// (cos picks sp on odd quadrants) and sign-flip table (cos flips on
// quadrants 1 and 2 → bit 1 of (j+1)). cos is even so input sign is
// discarded after the abs step. Owns its own kernel body.
TEXT ·cosVecSSE2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), DI
	MOVQ src_len+32(FP), BX

	CMPQ BX, $2
	JL   done_cos_sse2

	MOVSD  ·sincosAbsMask(SB), X10
	SHUFPD $0, X10, X10
	MOVSD  ·sincosFOPI(SB), X11
	SHUFPD $0, X11, X11
	MOVSD  ·sincosDP1(SB), X12
	SHUFPD $0, X12, X12
	MOVSD  ·sincosDP2(SB), X13
	SHUFPD $0, X13, X13
	MOVSD  ·sincosDP3(SB), X14
	SHUFPD $0, X14, X14

loop_cos_sse2:
	MOVUPD (DI), X0

	MOVAPD X0, X1
	ANDPD  X10, X1                   // X1 = |x|

	MOVAPD X1, X3
	MULPD  X11, X3                   // X3 = |x| * 2/pi
	CVTPD2PL X3, X4                  // X4 = j as 2x int32
	CVTPL2PD X4, X3                  // X3 = (double) j

	MOVAPD X3, X5
	MULPD  X12, X5
	SUBPD  X5, X1
	MOVAPD X3, X5
	MULPD  X13, X5
	SUBPD  X5, X1
	MOVAPD X3, X5
	MULPD  X14, X5
	SUBPD  X5, X1                    // X1 = y

	MOVAPD X1, X6
	MULPD  X1, X6                    // X6 = z

	// sin polynomial (sp)
	MOVSD  ·sincosS5(SB), X7
	SHUFPD $0, X7, X7
	MOVSD  ·sincosS4(SB), X8
	SHUFPD $0, X8, X8
	MULPD  X6, X7
	ADDPD  X8, X7
	MOVSD  ·sincosS3(SB), X8
	SHUFPD $0, X8, X8
	MULPD  X6, X7
	ADDPD  X8, X7
	MOVSD  ·sincosS2(SB), X8
	SHUFPD $0, X8, X8
	MULPD  X6, X7
	ADDPD  X8, X7
	MOVSD  ·sincosS1(SB), X8
	SHUFPD $0, X8, X8
	MULPD  X6, X7
	ADDPD  X8, X7
	MOVSD  ·sincosS0(SB), X8
	SHUFPD $0, X8, X8
	MULPD  X6, X7
	ADDPD  X8, X7
	MULPD  X6, X7
	MULPD  X1, X7
	ADDPD  X1, X7                    // X7 = sp

	// cos polynomial (cp)
	MOVSD  ·sincosC6(SB), X8
	SHUFPD $0, X8, X8
	MOVSD  ·sincosC5(SB), X9
	SHUFPD $0, X9, X9
	MULPD  X6, X8
	ADDPD  X9, X8
	MOVSD  ·sincosC4(SB), X9
	SHUFPD $0, X9, X9
	MULPD  X6, X8
	ADDPD  X9, X8
	MOVSD  ·sincosC3(SB), X9
	SHUFPD $0, X9, X9
	MULPD  X6, X8
	ADDPD  X9, X8
	MOVSD  ·sincosC2(SB), X9
	SHUFPD $0, X9, X9
	MULPD  X6, X8
	ADDPD  X9, X8
	MOVSD  ·sincosC1(SB), X9
	SHUFPD $0, X9, X9
	MULPD  X6, X8
	ADDPD  X9, X8
	MOVAPD X6, X9
	MULPD  X9, X9
	MULPD  X9, X8
	MOVSD  ·sincosHalf(SB), X9
	SHUFPD $0, X9, X9
	MULPD  X6, X9
	MOVSD  ·sincosOne(SB), X5
	SHUFPD $0, X5, X5
	SUBPD  X9, X5
	ADDPD  X5, X8                    // X8 = cp

	// Expand X4 (int32 j values) to int64x2 in X3, zero-extended.
	MOVDQA    X4, X3
	PXOR      X9, X9
	PUNPCKLLQ X9, X3                 // X3 = j zero-extended

	// Full-lane mask (0 or -1) from bit 0 of each j lane. PSLLQ alone
	// gives only bit 63 set, which is not enough for AND/OR blends —
	// emulating BLENDVPD bitwise needs every bit replicated.
	MOVQ       $1, DX
	MOVQ       DX, X12
	PUNPCKLQDQ X12, X12              // X12 = {1, 1}
	MOVDQA     X3, X11
	PAND       X12, X11              // X11 = (j & 1) per lane
	PXOR       X9, X9
	PSUBQ      X11, X9               // X9 = -(j & 1) per lane → 0 or all-ones

	// Cos picks sp when bit 0 of j is set
	MOVAPD X9, X5
	ANDPD  X9, X7                    // X7 = sp & mask
	ANDNPD X8, X5                    // X5 = (NOT mask) & cp
	ORPD   X7, X5                    // X5 = pick

	// Sign flip when bit 1 of (j+1) is set.
	// Build {1,1} as int64x2 in X9 and PADDQ to X3.
	MOVQ       $1, DX
	MOVQ       DX, X9                 // X9 lane 0 low 64 = 1, rest zero
	PUNPCKLQDQ X9, X9                 // duplicate: lane 0 = lane 1 = 1
	PADDQ      X9, X3                 // X3 = j + 1

	MOVDQA X3, X9
	PSLLQ  $62, X9                    // bit 1 of (j+1) at bit 63
	MOVSD  ·sincosSignBit(SB), X7
	SHUFPD $0, X7, X7
	ANDPD  X7, X9
	XORPD  X9, X5                     // apply sign

	MOVUPD X5, (AX)

	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_cos_sse2

done_cos_sse2:
	RET
