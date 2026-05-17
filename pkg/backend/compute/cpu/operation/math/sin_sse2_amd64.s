#include "textflag.h"

// sinVecSSE2(dst, src []float64) — 2 lanes per iteration, no FMA.
// Same Cody-Waite reduction + Cephes minimax polynomial as the AVX2
// path. CVTPD2PL rounds-to-nearest-even by default (MXCSR), so we use
// it in place of the missing VROUNDPD. No FMA: every fmadd lowers to
// MULPD + ADDPD / SUBPD. Owns its own kernel body.
TEXT ·sinVecSSE2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), DI
	MOVQ src_len+32(FP), BX

	CMPQ BX, $2
	JL   done_sin_sse2

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

loop_sin_sse2:
	MOVUPD (DI), X0                  // X0 = x

	MOVAPD X0, X1
	ANDPD  X10, X1                   // X1 = |x|

	MOVAPD X1, X2
	XORPD  X0, X2                    // X2 = sign(x)

	MOVAPD X1, X3
	MULPD  X11, X3                   // X3 = |x| * 2/pi
	CVTPD2PL X3, X4                  // X4 = j as 2x int32 (low half)
	CVTPL2PD X4, X3                  // X3 = (double) j

	MOVAPD X3, X5
	MULPD  X12, X5
	SUBPD  X5, X1                    // X1 = |x| - j*DP1
	MOVAPD X3, X5
	MULPD  X13, X5
	SUBPD  X5, X1                    // X1 -= j*DP2
	MOVAPD X3, X5
	MULPD  X14, X5
	SUBPD  X5, X1                    // X1 = y in [-pi/4, pi/4]

	MOVAPD X1, X6
	MULPD  X1, X6                    // X6 = z = y*y

	// sin polynomial: sp = y + y*z*P(z)
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
	ADDPD  X8, X7                    // X7 = P(z)
	MULPD  X6, X7                    // X7 = z*P(z)
	MULPD  X1, X7                    // X7 = y*z*P(z)
	ADDPD  X1, X7                    // X7 = sp

	// cos polynomial: cp = 1 - 0.5*z + z*z*Q(z)
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
	ADDPD  X9, X8                    // X8 = Q(z)
	MOVAPD X6, X9
	MULPD  X9, X9                    // X9 = z*z
	MULPD  X9, X8                    // X8 = z*z*Q(z)
	MOVSD  ·sincosHalf(SB), X9
	SHUFPD $0, X9, X9
	MULPD  X6, X9                    // X9 = 0.5*z
	MOVSD  ·sincosOne(SB), X5
	SHUFPD $0, X5, X5
	SUBPD  X9, X5                    // X5 = 1 - 0.5*z
	ADDPD  X5, X8                    // X8 = cp

	// Expand X4 (2 int32 j values) to int64x2 in X3, zero-extended.
	MOVDQA    X4, X3
	PXOR      X9, X9
	PUNPCKLLQ X9, X3                 // X3 = j zero-extended to int64x2

	// Build an all-ones / all-zeros mask from bit 0 of each lane.
	// Bitwise blends require a full-lane mask, not a single-bit-set
	// value — PSUBQ from zero of (j & 1) gives 0 or -1 per lane.
	MOVQ       $1, DX
	MOVQ       DX, X12
	PUNPCKLQDQ X12, X12              // X12 = {1, 1} int64x2
	MOVDQA     X3, X11
	PAND       X12, X11              // X11 = (j & 1) per lane (0 or 1)
	PXOR       X9, X9
	PSUBQ      X11, X9               // X9 = 0 - (j&1) = 0 or -1 per lane (full mask)

	// Emulated VBLENDVPD: result = (mask) ? cp : sp
	//                   = (cp AND mask) OR (sp AND NOT mask)
	MOVAPD X9, X5
	ANDPD  X9, X8                    // X8 = cp & mask
	ANDNPD X7, X5                    // X5 = (NOT mask) & sp
	ORPD   X8, X5                    // X5 = pick

	// Sign flip from bit 1 of j (PSLLQ 62 puts bit 1 at bit 63)
	MOVDQA X3, X9
	PSLLQ  $62, X9
	MOVSD  ·sincosSignBit(SB), X8
	SHUFPD $0, X8, X8
	ANDPD  X8, X9
	XORPD  X9, X2                    // combine with sign(x)
	XORPD  X2, X5                    // apply

	MOVUPD X5, (AX)

	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_sin_sse2

done_sin_sse2:
	RET
