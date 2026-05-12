#include "textflag.h"

// log(x):
//   bits = bitcast(x)
//   e = ((bits >> 52) & 0x7FF) - 1023
//   m_bits = (bits & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000  → m in [1,2)
//   if m > sqrt(2): m /= 2; e += 1
//   t = (m-1)/(m+1)   in [-0.172, 0.172]
//   log(m) = 2*t * P(t^2),  P(u) = 1 + u/3 + u^2/5 + ...
//   log(x) = log(m) + e*ln(2)

DATA ·logOne+0(SB)/8, $1.0
GLOBL ·logOne(SB), RODATA, $8
DATA ·logHalf+0(SB)/8, $0.5
GLOBL ·logHalf(SB), RODATA, $8
DATA ·logSqrt2+0(SB)/8, $1.4142135623730951
GLOBL ·logSqrt2(SB), RODATA, $8
DATA ·logLn2+0(SB)/8, $0.6931471805599453
GLOBL ·logLn2(SB), RODATA, $8
DATA ·logMantMask+0(SB)/8, $0x000FFFFFFFFFFFFF
GLOBL ·logMantMask(SB), RODATA, $8
DATA ·logBiasOne+0(SB)/8, $0x3FF0000000000000
GLOBL ·logBiasOne(SB), RODATA, $8
DATA ·logExpMask11+0(SB)/8, $0x7FF
GLOBL ·logExpMask11(SB), RODATA, $8
DATA ·logBias1023Q+0(SB)/8, $1023
GLOBL ·logBias1023Q(SB), RODATA, $8
DATA ·logMagic52+0(SB)/8, $0x4330000000000000
GLOBL ·logMagic52(SB), RODATA, $8
DATA ·logMagic52D+0(SB)/8, $4503599627370496.0
GLOBL ·logMagic52D(SB), RODATA, $8

DATA ·logA0+0(SB)/8, $1.0
GLOBL ·logA0(SB), RODATA, $8
DATA ·logA1+0(SB)/8, $0.3333333333333333
GLOBL ·logA1(SB), RODATA, $8
DATA ·logA2+0(SB)/8, $0.2
GLOBL ·logA2(SB), RODATA, $8
DATA ·logA3+0(SB)/8, $0.14285714285714285
GLOBL ·logA3(SB), RODATA, $8
DATA ·logA4+0(SB)/8, $0.1111111111111111
GLOBL ·logA4(SB), RODATA, $8
DATA ·logA5+0(SB)/8, $0.09090909090909091
GLOBL ·logA5(SB), RODATA, $8
DATA ·logA6+0(SB)/8, $0.07692307692307693
GLOBL ·logA6(SB), RODATA, $8

// logVecAVX2(dst, src []float64)
TEXT ·logVecAVX2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), DI
	MOVQ src_len+32(FP), BX

	VBROADCASTSD ·logOne(SB), Y10
	VBROADCASTSD ·logHalf(SB), Y11
	VBROADCASTSD ·logSqrt2(SB), Y12
	VBROADCASTSD ·logLn2(SB), Y13
	VBROADCASTSD ·logMantMask(SB), Y14
	VBROADCASTSD ·logBiasOne(SB), Y15

	CMPQ BX, $4
	JL   done_log_avx2
loop_log_avx2:
	VMOVUPD (DI), Y0                       // x

	// m = (bits & mantmask) | biasOne
	VANDPD Y14, Y0, Y3
	VORPD  Y15, Y3, Y3                     // Y3 = m in [1,2)

	// raw_exp = (bits >> 52) & 0x7FF
	VPSRLQ        $52, Y0, Y4
	VPBROADCASTQ  ·logExpMask11(SB), Y5
	VPAND         Y5, Y4, Y4               // Y4 = raw_exp (int64 lanes)
	VPBROADCASTQ  ·logBias1023Q(SB), Y5
	VPSUBQ        Y5, Y4, Y4               // Y4 = raw_exp - 1023 (signed)

	// Convert int64 in Y4 to double via add-2^52 / sub-2^52 magic.
	VPBROADCASTQ  ·logMagic52(SB), Y5
	VPADDQ        Y5, Y4, Y4
	VBROADCASTSD  ·logMagic52D(SB), Y5
	VSUBPD        Y5, Y4, Y4               // Y4 = e as double

	// If m > sqrt(2): m /= 2; e += 1
	VCMPPD    $14, Y12, Y3, Y5             // mask: m > sqrt2
	VMULPD    Y11, Y3, Y6                  // m*0.5
	VBLENDVPD Y5, Y6, Y3, Y3               // m = select(mask, m*0.5, m)
	VANDPD    Y10, Y5, Y6                  // 1.0 where mask
	VADDPD    Y6, Y4, Y4                   // e += correction

	// t = (m - 1) / (m + 1)
	VSUBPD Y10, Y3, Y6
	VADDPD Y10, Y3, Y7
	VDIVPD Y7, Y6, Y6
	VMULPD Y6, Y6, Y7                      // t^2

	// Horner: a6 + u*(a5 + u*(... + u*a0))
	VBROADCASTSD ·logA6(SB), Y2
	VBROADCASTSD ·logA5(SB), Y8
	VFMADD213PD Y8, Y7, Y2
	VBROADCASTSD ·logA4(SB), Y8
	VFMADD213PD Y8, Y7, Y2
	VBROADCASTSD ·logA3(SB), Y8
	VFMADD213PD Y8, Y7, Y2
	VBROADCASTSD ·logA2(SB), Y8
	VFMADD213PD Y8, Y7, Y2
	VBROADCASTSD ·logA1(SB), Y8
	VFMADD213PD Y8, Y7, Y2
	VBROADCASTSD ·logA0(SB), Y8
	VFMADD213PD Y8, Y7, Y2                 // Y2 = P(t^2)

	VMULPD Y6, Y2, Y2                      // t*P
	VADDPD Y2, Y2, Y2                      // 2*t*P = log(m)

	VFMADD231PD Y13, Y4, Y2                // log(m) + e*ln2

	VMOVUPD Y2, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_log_avx2

done_log_avx2:
	VZEROUPPER
	RET
