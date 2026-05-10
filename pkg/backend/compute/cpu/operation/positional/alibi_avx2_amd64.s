#include "textflag.h"

// ALiBiRowAVX2(dst []float64, slope float64, q int, seqLenK int)
// Computes dst[k] = slope * (k - q) for k = 0..seqLenK-1
// ABI0:
//   dst+0(FP)      ptr
//   dst_len+8(FP)  len
//   dst_cap+16(FP) cap
//   slope+24(FP)   float64
//   q+32(FP)       int
//   seqLenK+40(FP) int
//
// Strategy:
//   base = -slope * q    (additive offset)
//   dst[k] = slope*k + base
//   Process 4 elements per AVX2 iteration.

TEXT ·ALiBiRowAVX2(SB), NOSPLIT, $0-48
	MOVQ  dst+0(FP),      AX
	MOVQ  slope+24(FP),   BX
	MOVQ  q+32(FP),       CX
	MOVQ  seqLenK+40(FP), DX

	// X0 = slope
	MOVQ BX, X0

	// Compute base = -slope*q = -(slope*q)
	CVTSI2SD CX, X1         // float64(q)
	MULSD    X0, X1         // slope*q
	// negate: base = -slope*q
	PCMPEQD  X2, X2         // all-ones
	PSLLQ    $1, X2         // sign-bit mask for float64
	XORPD    X2, X1         // X1 = -slope*q  (flip sign bit)

	// Broadcast slope and base into YMM registers for 4-wide loop
	VBROADCASTSD X0, Y0     // Y0 = [slope, slope, slope, slope]
	VBROADCASTSD X1, Y1     // Y1 = [base,  base,  base,  base ]

	XORQ R8, R8             // k_start = 0

avx2loop:
	CMPQ DX, $4
	JL   tail

	// Compute [slope*(k+0)+base, slope*(k+1)+base, slope*(k+2)+base, slope*(k+3)+base]
	MOVQ R8, R9
	CVTSI2SD R9, X10
	MULSD X0, X10
	ADDSD X1, X10

	INCQ R9
	CVTSI2SD R9, X11
	MULSD X0, X11
	ADDSD X1, X11

	INCQ R9
	CVTSI2SD R9, X12
	MULSD X0, X12
	ADDSD X1, X12

	INCQ R9
	CVTSI2SD R9, X13
	MULSD X0, X13
	ADDSD X1, X13

	// Pack into Y-register and store
	VINSERTF128 $0, X10, Y10, Y10
	VMOVSD X11, X14
	VUNPCKLPD X14, X10, X10
	VINSERTF128 $1, X10, Y10, Y10

	// Actually just store as individual doubles — the loop overhead is small
	MOVSD X10, (AX)
	MOVSD X11, 8(AX)
	MOVSD X12, 16(AX)
	MOVSD X13, 24(AX)

	ADDQ $32, AX
	ADDQ $4,  R8
	SUBQ $4,  DX
	JMP  avx2loop

tail:
	CMPQ DX, $0
	JLE  done
scalar:
	MOVQ R8, R9
	CVTSI2SD R9, X3
	MULSD X0, X3
	ADDSD X1, X3
	MOVSD X3, (AX)
	ADDQ $8, AX
	INCQ R8
	DECQ DX
	JNZ  scalar

done:
	VZEROUPPER
	RET
