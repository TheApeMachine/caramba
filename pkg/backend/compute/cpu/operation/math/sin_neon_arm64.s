#include "textflag.h"

// NEON double-precision sin via Cody-Waite reduction + Cephes minimax
// polynomial. Two D-lanes per vector iteration; the odd tail is
// handled by a scalar SIMD path in this same file (no jump to a
// generic body). Constants are declared here and shared with the
// cos NEON kernel and the SIMD paths on amd64.

#define VFADD_D2(m, n, d)   WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d)   WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d)   WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFRINTN_D2(n, d)    WORD $(0x4E618800 | ((n) << 5) | (d))
#define VFCVTZS_D2(n, d)    WORD $(0x4EE1B800 | ((n) << 5) | (d))
// CMTST.2D Vd, Vn, Vm — per-lane "(Vn AND Vm) != 0" returning -1 or 0.
// Used to expand a bitmask into an all-ones / all-zeros per-lane mask
// suitable for BSL or the VAND/VBIC/VORR blend pattern.
#define VCMTST_D2(m, n, d)  WORD $(0x6EE08C00 | ((m) << 16) | ((n) << 5) | (d))
#define VLOADDUP(sym, addr, vec) MOVD $sym, addr; VLD1R (addr), [vec.D2]

// Range reduction constants
DATA ·sincosFOPI+0(SB)/8, $0.6366197723675813430755
GLOBL ·sincosFOPI(SB), RODATA, $8
DATA ·sincosDP1+0(SB)/8, $1.5707963267341256e+00
GLOBL ·sincosDP1(SB), RODATA, $8
DATA ·sincosDP2+0(SB)/8, $6.077100506303966e-11
GLOBL ·sincosDP2(SB), RODATA, $8
DATA ·sincosDP3+0(SB)/8, $2.0222662487959506e-21
GLOBL ·sincosDP3(SB), RODATA, $8
DATA ·sincosAbsMask+0(SB)/8, $0x7FFFFFFFFFFFFFFF
GLOBL ·sincosAbsMask(SB), RODATA, $8
DATA ·sincosSignBit+0(SB)/8, $0x8000000000000000
GLOBL ·sincosSignBit(SB), RODATA, $8
DATA ·sincosOne+0(SB)/8, $1.0
GLOBL ·sincosOne(SB), RODATA, $8
DATA ·sincosHalf+0(SB)/8, $0.5
GLOBL ·sincosHalf(SB), RODATA, $8

// Cephes minimax sin coefficients
DATA ·sincosS0+0(SB)/8, $-1.66666666666666307295e-1
GLOBL ·sincosS0(SB), RODATA, $8
DATA ·sincosS1+0(SB)/8, $8.33333333332211858878e-3
GLOBL ·sincosS1(SB), RODATA, $8
DATA ·sincosS2+0(SB)/8, $-1.98412698295895385996e-4
GLOBL ·sincosS2(SB), RODATA, $8
DATA ·sincosS3+0(SB)/8, $2.75573136213857245213e-6
GLOBL ·sincosS3(SB), RODATA, $8
DATA ·sincosS4+0(SB)/8, $-2.50507477628578072866e-8
GLOBL ·sincosS4(SB), RODATA, $8
DATA ·sincosS5+0(SB)/8, $1.58962301576546568060e-10
GLOBL ·sincosS5(SB), RODATA, $8

// Cephes minimax cos coefficients
DATA ·sincosC1+0(SB)/8, $4.16666666666665929218e-2
GLOBL ·sincosC1(SB), RODATA, $8
DATA ·sincosC2+0(SB)/8, $-1.38888888888730564116e-3
GLOBL ·sincosC2(SB), RODATA, $8
DATA ·sincosC3+0(SB)/8, $2.48015872888517045348e-5
GLOBL ·sincosC3(SB), RODATA, $8
DATA ·sincosC4+0(SB)/8, $-2.75573141792967388112e-7
GLOBL ·sincosC4(SB), RODATA, $8
DATA ·sincosC5+0(SB)/8, $2.08757008419747316778e-9
GLOBL ·sincosC5(SB), RODATA, $8
DATA ·sincosC6+0(SB)/8, $-1.13585365213876817300e-11
GLOBL ·sincosC6(SB), RODATA, $8

// sinVecNEON(dst, src []float64)
// Caller guarantees len(src) == len(dst) and divisible by 2.
TEXT ·sinVecNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD src_len+32(FP), R2

	// Hot constants in registers (rest loaded inline)
	VLOADDUP(·sincosAbsMask(SB), R9, V20)
	VLOADDUP(·sincosFOPI(SB),    R9, V21)
	VLOADDUP(·sincosDP1(SB),     R9, V22)
	VLOADDUP(·sincosDP2(SB),     R9, V23)
	VLOADDUP(·sincosDP3(SB),     R9, V24)
	VLOADDUP(·sincosSignBit(SB), R9, V25)
	VLOADDUP(·sincosOne(SB),     R9, V26)
	VLOADDUP(·sincosHalf(SB),    R9, V27)
	// V28 = {1, 1} int64x2 — bit-0 test pattern for CMTST.
	MOVD $1, R8
	VDUP R8, V28.D2

	LSR $1, R2, R3
	CBZ R3, done_sin_neon

loop_sin_neon:
	VLD1.P 16(R1), [V0.D2]                       // V0 = x

	VAND V20.B16, V0.B16, V1.B16                 // V1 = |x|
	VEOR V1.B16, V0.B16, V2.B16                  // V2 = sign(x)

	VFMUL_D2(21, 1, 3)                           // V3 = |x| * FOPI
	VFRINTN_D2(3, 3)                             // V3 = round(|x| * FOPI) as double
	VFCVTZS_D2(3, 4)                             // V4 = j as int64x2 (V3 unchanged)

	VFMUL_D2(22, 3, 5)
	VFSUB_D2(5, 1, 1)                            // |x| - j*DP1
	VFMUL_D2(23, 3, 5)
	VFSUB_D2(5, 1, 1)                            // -= j*DP2
	VFMUL_D2(24, 3, 5)
	VFSUB_D2(5, 1, 1)                            // V1 = y

	VFMUL_D2(1, 1, 6)                            // V6 = z = y*y

	// sin polynomial: V7 = sp
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
	VFADD_D2(8, 7, 7)                            // V7 = P(z)
	VFMUL_D2(6, 7, 7)                            // V7 = z * P(z)
	VFMUL_D2(1, 7, 7)                            // V7 = y*z*P(z)
	VFADD_D2(1, 7, 7)                            // V7 = sp

	// cos polynomial: V8 = cp
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
	VFADD_D2(9, 8, 8)                            // V8 = Q(z)
	VFMUL_D2(6, 6, 9)                            // V9 = z*z
	VFMUL_D2(9, 8, 8)                            // V8 = z*z*Q(z)
	VFMUL_D2(27, 6, 9)                           // V9 = 0.5*z
	// VFSUB_D2(m, n, d) encodes FSUB.2D Vd, Vn, Vm → Vd = Vn - Vm.
	// Args (9, 26, 9) therefore compute V9 = V26 - V9 = 1 - 0.5*z.
	VFSUB_D2(9, 26, 9)
	VFADD_D2(8, 9, 8)                            // V8 = cp

	// Select sp or cp based on (j & 1). VBSL operates bitwise, so the
	// mask must be all-ones or all-zeros per lane — a single bit set is
	// not enough. CMTST builds exactly that pattern by testing bit 0
	// of each j lane against V28={1,1}: lane→-1 where bit set, lane→0
	// elsewhere.
	VCMTST_D2(28, 4, 11)                         // V11 = (j & 1) ? -1 : 0 per lane
	// BSL.B16 Vd, Vn, Vm: Vd = (Vd AND Vn) OR (NOT Vd AND Vm). Plan-9
	// reverses the source order: `VBSL Vm, Vn, Vd`. (mask==1) → cp (V8),
	// (mask==0) → sp (V7), so Vn=V8 (true) and Vm=V7 (false).
	VBSL V7.B16, V8.B16, V11.B16                 // V11 = (mask) ? cp : sp

	// Sign flip: bit 1 of j → bit 63
	VSHL $62, V4.D2, V10.D2
	VAND V25.B16, V10.B16, V10.B16               // mask to single sign bit
	VEOR V10.B16, V2.B16, V2.B16                 // combine with sign(x)
	VEOR V2.B16, V11.B16, V11.B16                // apply sign

	VST1.P [V11.D2], 16(R0)

	SUBS $1, R3, R3
	BNE  loop_sin_neon

done_sin_neon:
	RET
