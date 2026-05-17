#include "textflag.h"

// Vectorized double-precision sin via Cody-Waite range reduction plus
// minimax polynomial. Cephes algorithm:
//   j  = round(|x| * 2/pi)               // quadrant count
//   y  = ((|x| - j*DP1) - j*DP2) - j*DP3 // reduced to [-pi/4, pi/4]
//   z  = y*y
//   sp = y + y*z * (S0 + z*(S1 + z*(S2 + z*(S3 + z*(S4 + z*S5)))))
//   cp = 1 - 0.5*z + z*z * (C1 + z*(C2 + z*(C3 + z*(C4 + z*(C5 + z*C6)))))
//   pick = (j & 1) ? cp : sp
//   sign = sign(x) ^ ((j>>1) & 1 → sign bit)
//   result = pick xor sign-bit
//
// Constants defined here are shared with sin_sse2_amd64.s and
// cos_*_amd64.s (Go assembler resolves the same name across .s files
// in one package). Each .s file owns its own kernel body; no aliasing.

// Range reduction constants ---------------------------------------------------
DATA ·sincosFOPI+0(SB)/8, $0.6366197723675813430755
GLOBL ·sincosFOPI(SB), RODATA, $8
DATA ·sincosDP1+0(SB)/8, $1.5707963267341256e+00
GLOBL ·sincosDP1(SB), RODATA, $8
DATA ·sincosDP2+0(SB)/8, $6.077100506303966e-11
GLOBL ·sincosDP2(SB), RODATA, $8
DATA ·sincosDP3+0(SB)/8, $2.0222662487959506e-21
GLOBL ·sincosDP3(SB), RODATA, $8

// Bit masks ------------------------------------------------------------------
DATA ·sincosAbsMask+0(SB)/8, $0x7FFFFFFFFFFFFFFF
GLOBL ·sincosAbsMask(SB), RODATA, $8
DATA ·sincosSignBit+0(SB)/8, $0x8000000000000000
GLOBL ·sincosSignBit(SB), RODATA, $8

DATA ·sincosOne+0(SB)/8, $1.0
GLOBL ·sincosOne(SB), RODATA, $8
DATA ·sincosHalf+0(SB)/8, $0.5
GLOBL ·sincosHalf(SB), RODATA, $8

// Cephes minimax polynomial coefficients (valid in [-pi/4, pi/4]) -------------
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

// sinVecAVX2(dst, src []float64)
// Processes 4 lanes per iteration using AVX2 + FMA. Caller guarantees
// len(src) and len(dst) are equal and divisible by 4.
TEXT ·sinVecAVX2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), DI
	MOVQ src_len+32(FP), BX

	VBROADCASTSD ·sincosFOPI(SB), Y10
	VBROADCASTSD ·sincosDP1(SB), Y11
	VBROADCASTSD ·sincosDP2(SB), Y12
	VBROADCASTSD ·sincosDP3(SB), Y13
	VBROADCASTSD ·sincosAbsMask(SB), Y14

	CMPQ BX, $4
	JL   done_sin_avx2

loop_sin_avx2:
	VMOVUPD (DI), Y0                 // Y0 = x

	VANDPD       Y14, Y0, Y1         // Y1 = |x|
	VXORPD       Y0, Y1, Y2          // Y2 = sign bit of x (in bit 63 of each lane)

	VMULPD       Y10, Y1, Y3         // Y3 = |x| * 2/pi
	VROUNDPD     $0, Y3, Y3          // Y3 = round(|x| * 2/pi)

	VCVTPD2DQY   Y3, X4              // X4 = j as 4x int32

	VFNMADD231PD Y11, Y3, Y1         // Y1 = |x| - j*DP1
	VFNMADD231PD Y12, Y3, Y1         // Y1 -= j*DP2
	VFNMADD231PD Y13, Y3, Y1         // Y1 -= j*DP3  -> y in [-pi/4, pi/4]

	VMULPD       Y1, Y1, Y6          // Y6 = z = y*y

	// sin polynomial: sp = y + y*z*(S0 + z*(S1 + z*(S2 + z*(S3 + z*(S4 + z*S5)))))
	VBROADCASTSD ·sincosS5(SB), Y7
	VBROADCASTSD ·sincosS4(SB), Y8
	VFMADD213PD  Y8, Y6, Y7          // Y7 = Y7*z + S4
	VBROADCASTSD ·sincosS3(SB), Y8
	VFMADD213PD  Y8, Y6, Y7
	VBROADCASTSD ·sincosS2(SB), Y8
	VFMADD213PD  Y8, Y6, Y7
	VBROADCASTSD ·sincosS1(SB), Y8
	VFMADD213PD  Y8, Y6, Y7
	VBROADCASTSD ·sincosS0(SB), Y8
	VFMADD213PD  Y8, Y6, Y7          // Y7 = sin polynomial value P(z)
	VMULPD       Y6, Y7, Y7          // Y7 = z * P(z)
	VFMADD213PD  Y1, Y1, Y7          // Y7 = y * (z*P(z)) + y  = sp

	// cos polynomial: cp = 1 - 0.5*z + z*z*(C1 + z*(C2 + z*(C3 + z*(C4 + z*(C5 + z*C6)))))
	VBROADCASTSD ·sincosC6(SB), Y8
	VBROADCASTSD ·sincosC5(SB), Y9
	VFMADD213PD  Y9, Y6, Y8          // Y8 = Y8*z + C5
	VBROADCASTSD ·sincosC4(SB), Y9
	VFMADD213PD  Y9, Y6, Y8
	VBROADCASTSD ·sincosC3(SB), Y9
	VFMADD213PD  Y9, Y6, Y8
	VBROADCASTSD ·sincosC2(SB), Y9
	VFMADD213PD  Y9, Y6, Y8
	VBROADCASTSD ·sincosC1(SB), Y9
	VFMADD213PD  Y9, Y6, Y8          // Y8 = Q(z)
	VMULPD       Y6, Y6, Y9          // Y9 = z*z
	VMULPD       Y9, Y8, Y8          // Y8 = z*z * Q(z)
	VBROADCASTSD ·sincosHalf(SB), Y9
	VMULPD       Y6, Y9, Y9          // Y9 = 0.5*z
	VBROADCASTSD ·sincosOne(SB), Y15
	VSUBPD       Y9, Y15, Y9         // Y9 = 1 - 0.5*z
	VADDPD       Y8, Y9, Y8          // Y8 = cp

	// Select sp or cp based on (j & 1)
	// Build mask: high bit of each int64 lane = bit 0 of j
	VPMOVSXDQ    X4, Y15             // Y15 = j as int64x4
	VPSLLQ       $63, Y15, Y9        // Y9 = (j & 1) << 63 → mask via sign bit
	VBLENDVPD    Y9, Y8, Y7, Y7      // Y7 = (mask) ? cp : sp

	// Sign flip based on bit 1 of j: high bit if (j>>1)&1
	VPSLLQ       $62, Y15, Y9        // Y9 = bit 1 of j shifted to bit 63
	VBROADCASTSD ·sincosSignBit(SB), Y8
	VANDPD       Y8, Y9, Y9          // keep only sign bit
	VXORPD       Y9, Y2, Y2          // combine with original sign of x
	VXORPD       Y2, Y7, Y7          // final sign applied

	VMOVUPD      Y7, (AX)

	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_sin_avx2

done_sin_avx2:
	VZEROUPPER
	RET
