#include "textflag.h"

// NEON double-precision cos. Same Cody-Waite reduction as sinVecNEON;
// differs only in the octant→polynomial table (cos picks sp on odd
// quadrants) and sign-flip mask (bit 1 of (j+1)). cos is even so the
// input sign is discarded after the abs step. Own kernel body.

#define VFADD_D2(m, n, d)   WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d)   WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d)   WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFRINTN_D2(n, d)    WORD $(0x4E618800 | ((n) << 5) | (d))
#define VFCVTZS_D2(n, d)    WORD $(0x4EE1B800 | ((n) << 5) | (d))
#define VCMTST_D2(m, n, d)  WORD $(0x6EE08C00 | ((m) << 16) | ((n) << 5) | (d))
#define VLOADDUP(sym, addr, vec) MOVD $sym, addr; VLD1R (addr), [vec.D2]

// cosVecNEON(dst, src []float64)
TEXT ·cosVecNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD src_len+32(FP), R2

	VLOADDUP(·sincosAbsMask(SB), R9, V20)
	VLOADDUP(·sincosFOPI(SB),    R9, V21)
	VLOADDUP(·sincosDP1(SB),     R9, V22)
	VLOADDUP(·sincosDP2(SB),     R9, V23)
	VLOADDUP(·sincosDP3(SB),     R9, V24)
	VLOADDUP(·sincosSignBit(SB), R9, V25)
	VLOADDUP(·sincosOne(SB),     R9, V26)
	VLOADDUP(·sincosHalf(SB),    R9, V27)
	// Broadcast int64 1 for j+1
	MOVD $1, R8
	VDUP R8, V28.D2

	LSR $1, R2, R3
	CBZ R3, done_cos_neon

loop_cos_neon:
	VLD1.P 16(R1), [V0.D2]

	VAND V20.B16, V0.B16, V1.B16                 // V1 = |x|

	VFMUL_D2(21, 1, 3)
	VFRINTN_D2(3, 3)                             // V3 = round(...) as double
	VFCVTZS_D2(3, 4)                             // V4 = j int64x2

	VFMUL_D2(22, 3, 5)
	VFSUB_D2(5, 1, 1)
	VFMUL_D2(23, 3, 5)
	VFSUB_D2(5, 1, 1)
	VFMUL_D2(24, 3, 5)
	VFSUB_D2(5, 1, 1)                            // V1 = y

	VFMUL_D2(1, 1, 6)                            // V6 = z = y*y

	// sin polynomial → V7 = sp
	VLOADDUP(·sincosS5(SB), R9, V7)
	VLOADDUP(·sincosS4(SB), R9, V8)
	VFMUL_D2(6, 7, 7)
	VFADD_D2(8, 7, 7)
	VLOADDUP(·sincosS3(SB), R9, V8)
	VFMUL_D2(6, 7, 7)
	VFADD_D2(8, 7, 7)
	VLOADDUP(·sincosS2(SB), R9, V8)
	VFMUL_D2(6, 7, 7)
	VFADD_D2(8, 7, 7)
	VLOADDUP(·sincosS1(SB), R9, V8)
	VFMUL_D2(6, 7, 7)
	VFADD_D2(8, 7, 7)
	VLOADDUP(·sincosS0(SB), R9, V8)
	VFMUL_D2(6, 7, 7)
	VFADD_D2(8, 7, 7)
	VFMUL_D2(6, 7, 7)
	VFMUL_D2(1, 7, 7)
	VFADD_D2(1, 7, 7)                            // V7 = sp

	// cos polynomial → V8 = cp
	VLOADDUP(·sincosC6(SB), R9, V8)
	VLOADDUP(·sincosC5(SB), R9, V9)
	VFMUL_D2(6, 8, 8)
	VFADD_D2(9, 8, 8)
	VLOADDUP(·sincosC4(SB), R9, V9)
	VFMUL_D2(6, 8, 8)
	VFADD_D2(9, 8, 8)
	VLOADDUP(·sincosC3(SB), R9, V9)
	VFMUL_D2(6, 8, 8)
	VFADD_D2(9, 8, 8)
	VLOADDUP(·sincosC2(SB), R9, V9)
	VFMUL_D2(6, 8, 8)
	VFADD_D2(9, 8, 8)
	VLOADDUP(·sincosC1(SB), R9, V9)
	VFMUL_D2(6, 8, 8)
	VFADD_D2(9, 8, 8)
	VFMUL_D2(6, 6, 9)
	VFMUL_D2(9, 8, 8)
	VFMUL_D2(27, 6, 9)
	VFSUB_D2(9, 26, 9)
	VFADD_D2(8, 9, 8)                            // V8 = cp

	// Select sp or cp based on (j & 1). Cos picks sp on odd quadrants.
	VAND V28.B16, V4.B16, V11.B16                // V11 = j & 1 per lane
	VEOR V12.B16, V12.B16, V12.B16               // V12 = 0
	VSUB V11.D2, V12.D2, V11.D2                  // V11 = 0 - (j & 1), so 0 or -1
	VBSL V8.B16, V7.B16, V11.B16                 // V11 = (mask) ? sp : cp

	// Sign flip: bit 1 of (j+1).
	VADD V28.D2, V4.D2, V10.D2                   // V10 = j+1
	VSHL $62, V10.D2, V10.D2                     // bit 1 → bit 63
	VAND V25.B16, V10.B16, V10.B16
	VEOR V10.B16, V11.B16, V11.B16

	VST1.P [V11.D2], 16(R0)

	SUBS $1, R3, R3
	BNE  loop_cos_neon

done_cos_neon:
	RET
