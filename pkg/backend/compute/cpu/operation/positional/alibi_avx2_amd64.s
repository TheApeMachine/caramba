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
DATA ·alibiOffset4+0(SB)/8, $0.0
DATA ·alibiOffset4+8(SB)/8, $1.0
DATA ·alibiOffset4+16(SB)/8, $2.0
DATA ·alibiOffset4+24(SB)/8, $3.0
GLOBL ·alibiOffset4(SB), RODATA, $32

DATA ·alibiStep4+0(SB)/8, $4.0
GLOBL ·alibiStep4(SB), RODATA, $8

TEXT ·ALiBiRowAVX2(SB), NOSPLIT, $0-48
	MOVQ  dst+0(FP),      AX
	MOVQ  slope+24(FP),   BX
	MOVQ  q+32(FP),       CX
	MOVQ  seqLenK+40(FP), DX

	MOVQ BX, X0                    // slope

	CVTSQ2SD CX, X1                // float64(q)
	MULSD    X0, X1                // slope*q
	XORPD    X2, X2
	SUBSD    X1, X2                // -slope*q

	VBROADCASTSD X0, Y0            // slope
	VBROADCASTSD X2, Y1            // base
	VMOVUPD ·alibiOffset4(SB), Y2  // [0, 1, 2, 3]
	VBROADCASTSD ·alibiStep4(SB), Y3

avx2loop:
	CMPQ DX, $4
	JL   tail

	VMULPD Y0, Y2, Y4
	VADDPD Y1, Y4, Y4
	VMOVUPD Y4, (AX)
	VADDPD Y3, Y2, Y2

	ADDQ $32, AX
	SUBQ $4,  DX
	JMP  avx2loop

tail:
	CMPQ DX, $0
	JLE  done
	MOVQ seqLenK+40(FP), R8
	SUBQ DX, R8
scalar:
	MOVQ R8, R9
	CVTSQ2SD R9, X3
	MULSD X0, X3
	ADDSD X2, X3
	MOVSD X3, (AX)
	ADDQ $8, AX
	INCQ R8
	DECQ DX
	JNZ  scalar

done:
	VZEROUPPER
	RET
