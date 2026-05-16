#include "textflag.h"

// ropeAdvanceRowAVX2(cosCur, sinCur, cosStep, sinStep []float64)
// For all i in 0..len:
//   cosNext[i] = cosCur[i]*cosStep[i] - sinCur[i]*sinStep[i]
//   sinNext[i] = sinCur[i]*cosStep[i] + cosCur[i]*sinStep[i]
// Result overwrites cosCur and sinCur in place — caller copies to next row.
TEXT ·ropeAdvanceRowAVX2(SB), NOSPLIT, $0-96
	MOVQ cosCur+0(FP), AX
	MOVQ sinCur+24(FP), BX
	MOVQ cosStep+48(FP), DI
	MOVQ sinStep+72(FP), SI
	MOVQ cosCur_len+8(FP), CX
	CMPQ CX, $4
	JL ra_tail
ra_loop:
	VMOVUPD (AX), Y0                          // cosCur
	VMOVUPD (BX), Y1                          // sinCur
	VMOVUPD (DI), Y2                          // cosStep
	VMOVUPD (SI), Y3                          // sinStep
	VMULPD Y2, Y0, Y4                         // cosCur*cosStep
	VFNMADD231PD Y3, Y1, Y4                   // -sinCur*sinStep
	VMULPD Y2, Y1, Y5                         // sinCur*cosStep
	VFMADD231PD Y3, Y0, Y5                    // +cosCur*sinStep
	VMOVUPD Y4, (AX)
	VMOVUPD Y5, (BX)
	ADDQ $32, AX
	ADDQ $32, BX
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $4, CX
	CMPQ CX, $4
	JGE ra_loop
ra_tail:
	CMPQ CX, $2
	JL ra_scalar
	MOVUPD (AX), X0
	MOVUPD (BX), X1
	MOVUPD (DI), X2
	MOVUPD (SI), X3
	MOVAPD X0, X4
	MULPD X2, X4
	MOVAPD X1, X6
	MULPD X3, X6
	SUBPD X6, X4
	MOVAPD X1, X5
	MULPD X2, X5
	MOVAPD X0, X6
	MULPD X3, X6
	ADDPD X6, X5
	MOVUPD X4, (AX)
	MOVUPD X5, (BX)
	ADDQ $16, AX
	ADDQ $16, BX
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, CX
ra_scalar:
	CMPQ CX, $0
	JLE ra_done
	MOVSD (AX), X0
	MOVSD (BX), X1
	MOVSD (DI), X2
	MOVSD (SI), X3
	MOVAPD X0, X4
	MULSD X2, X4
	MOVAPD X1, X6
	MULSD X3, X6
	SUBSD X6, X4
	MOVAPD X1, X5
	MULSD X2, X5
	MOVAPD X0, X6
	MULSD X3, X6
	ADDSD X6, X5
	MOVSD X4, (AX)
	MOVSD X5, (BX)
ra_done:
	VZEROUPPER
	RET
