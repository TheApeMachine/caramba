#include "textflag.h"

// Vectorized double-precision cos via Cody-Waite range reduction plus
// minimax polynomial. Same reduction as sin; differs only in the
// octant-to-polynomial mapping:
//   pick = (j & 1) ? sp : cp
//   sign-flip when bit 1 of (j+1) is set
// cos is even, so the input sign is discarded.
//
// Shares sincos constants declared by sin_avx2_amd64.s. Kernel body
// is its own — no aliasing or JMP into sin.

// cosVecAVX2(dst, src []float64)
TEXT ·cosVecAVX2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), DI
	MOVQ src_len+32(FP), BX

	VBROADCASTSD ·sincosFOPI(SB), Y10
	VBROADCASTSD ·sincosDP1(SB), Y11
	VBROADCASTSD ·sincosDP2(SB), Y12
	VBROADCASTSD ·sincosDP3(SB), Y13
	VBROADCASTSD ·sincosAbsMask(SB), Y14

	CMPQ BX, $4
	JL   done_cos_avx2

loop_cos_avx2:
	VMOVUPD (DI), Y0

	VANDPD       Y14, Y0, Y1         // Y1 = |x|

	VMULPD       Y10, Y1, Y3         // Y3 = |x| * 2/pi
	VROUNDPD     $0, Y3, Y3          // Y3 = round(|x| * 2/pi)

	VCVTPD2DQY   Y3, X4              // X4 = j as 4x int32

	VFNMADD231PD Y11, Y3, Y1         // Y1 = |x| - j*DP1
	VFNMADD231PD Y12, Y3, Y1         // -= j*DP2
	VFNMADD231PD Y13, Y3, Y1         // -= j*DP3  -> y

	VMULPD       Y1, Y1, Y6          // Y6 = z = y*y

	// sin polynomial (sp = y + y*z*P(z))
	VBROADCASTSD ·sincosS5(SB), Y7
	VBROADCASTSD ·sincosS4(SB), Y8
	VFMADD213PD  Y8, Y6, Y7
	VBROADCASTSD ·sincosS3(SB), Y8
	VFMADD213PD  Y8, Y6, Y7
	VBROADCASTSD ·sincosS2(SB), Y8
	VFMADD213PD  Y8, Y6, Y7
	VBROADCASTSD ·sincosS1(SB), Y8
	VFMADD213PD  Y8, Y6, Y7
	VBROADCASTSD ·sincosS0(SB), Y8
	VFMADD213PD  Y8, Y6, Y7
	VMULPD       Y6, Y7, Y7
	VFMADD213PD  Y1, Y1, Y7          // Y7 = sp

	// cos polynomial (cp = 1 - 0.5*z + z*z*Q(z))
	VBROADCASTSD ·sincosC6(SB), Y8
	VBROADCASTSD ·sincosC5(SB), Y9
	VFMADD213PD  Y9, Y6, Y8
	VBROADCASTSD ·sincosC4(SB), Y9
	VFMADD213PD  Y9, Y6, Y8
	VBROADCASTSD ·sincosC3(SB), Y9
	VFMADD213PD  Y9, Y6, Y8
	VBROADCASTSD ·sincosC2(SB), Y9
	VFMADD213PD  Y9, Y6, Y8
	VBROADCASTSD ·sincosC1(SB), Y9
	VFMADD213PD  Y9, Y6, Y8
	VMULPD       Y6, Y6, Y9          // Y9 = z*z
	VMULPD       Y9, Y8, Y8          // Y8 = z*z * Q(z)
	VBROADCASTSD ·sincosHalf(SB), Y9
	VMULPD       Y6, Y9, Y9
	VBROADCASTSD ·sincosOne(SB), Y15
	VSUBPD       Y9, Y15, Y9
	VADDPD       Y8, Y9, Y8          // Y8 = cp

	// Select sp or cp based on (j & 1) — cos picks SP when bit 0 is set
	VPMOVSXDQ    X4, Y15             // Y15 = j as int64x4
	VPSLLQ       $63, Y15, Y9        // Y9 = (j & 1) → mask sign bit
	VBLENDVPD    Y9, Y7, Y8, Y8      // Y8 = (mask) ? sp : cp

	// Sign flip: bit 1 of (j+1). Compute (j+1) as int64x4 first.
	MOVQ         $1, DX
	MOVQ         DX, X5
	VPBROADCASTQ X5, Y5              // Y5 = {1,1,1,1}
	VPADDQ       Y5, Y15, Y15        // Y15 = j+1
	VPSLLQ       $62, Y15, Y9        // Y9 = bit 1 of (j+1) at bit 63
	VBROADCASTSD ·sincosSignBit(SB), Y5
	VANDPD       Y5, Y9, Y9          // keep only sign bit
	VXORPD       Y9, Y8, Y8          // apply sign

	VMOVUPD      Y8, (AX)

	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_cos_avx2

done_cos_avx2:
	VZEROUPPER
	RET
