#include "textflag.h"

// ALiBiRowSSE2(dst []float64, slope float64, q int, seqLenK int)
// Computes dst[k] = slope * (k - q) two lanes at a time.
// ABI0: dst+0, slope+24, q+32, seqLenK+40
DATA ·alibiOffset2+0(SB)/8, $0.0
DATA ·alibiOffset2+8(SB)/8, $1.0
GLOBL ·alibiOffset2(SB), RODATA, $16

DATA ·alibiStep2+0(SB)/8, $2.0
DATA ·alibiStep2+8(SB)/8, $2.0
GLOBL ·alibiStep2(SB), RODATA, $16

TEXT ·ALiBiRowSSE2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP),      AX
	MOVQ slope+24(FP),   BX
	MOVQ q+32(FP),       CX
	MOVQ seqLenK+40(FP), DX

	MOVQ BX, X0                 // slope
	MOVAPD X0, X4
	SHUFPD $0, X4, X4           // [slope, slope]

	CVTSQ2SD CX, X1             // float64(q)
	MULSD X0, X1                // slope*q
	XORPD X2, X2
	SUBSD X1, X2                // -slope*q
	MOVAPD X2, X5
	SHUFPD $0, X5, X5           // [base, base]

	MOVUPD ·alibiOffset2(SB), X6
	MOVUPD ·alibiStep2(SB), X7

	XORQ R8, R8                 // k = 0
loop:
	CMPQ DX, $2
	JL   tail

	MOVAPD X6, X3
	MULPD  X4, X3
	ADDPD  X5, X3
	MOVUPD X3, (AX)
	ADDPD  X7, X6

	ADDQ $16, AX
	ADDQ $2, R8
	SUBQ $2, DX
	JMP  loop

tail:
	CMPQ DX, $0
	JLE  done

	MOVQ R8, R9
	CVTSQ2SD R9, X3
	MULSD X0, X3
	ADDSD X2, X3
	MOVSD X3, (AX)
done:
	RET
