#include "textflag.h"

// freeEnergySSE2(mu, expSigma []float64) float64
// ABI0: mu+0(FP)..16, expSigma+24(FP)..40, ret+48(FP)
TEXT ·freeEnergySSE2(SB), NOSPLIT, $0-56
	MOVQ mu+0(FP),        AX
	MOVQ mu_len+8(FP),    BX
	MOVQ expSigma+24(FP), DI
	XORPD X0, X0

	CMPQ BX, $2
	JL   fe2_scalar

fe2_loop2:
	MOVUPD (AX), X1
	MOVUPD (DI), X2
	MULPD  X1, X1
	ADDPD  X1, X0
	ADDPD  X2, X0
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  fe2_loop2

	MOVAPD X0, X1
	UNPCKHPD X0, X1
	ADDSD X1, X0

fe2_scalar:
	TESTQ BX, BX
	JZ    fe2_done
	MOVSD (AX), X1
	MOVSD (DI), X2
	MULSD X1, X1
	ADDSD X1, X0
	ADDSD X2, X0

fe2_done:
	MOVSD X0, ret+48(FP)
	RET

// beliefUpdateMuSSE2(dst, mu, predErr []float64, lr float64)
// ABI0: dst+0(FP)..16, mu+24(FP)..40, predErr+48(FP)..64, lr+72(FP)
TEXT ·beliefUpdateMuSSE2(SB), NOSPLIT, $0-80
	MOVQ dst+0(FP),      R8
	MOVQ mu+24(FP),      R9
	MOVQ mu_len+32(FP),  BX
	MOVQ predErr+48(FP), R10
	MOVSD lr+72(FP), X14
	SHUFPD $0, X14, X14

	CMPQ BX, $2
	JL   bmu2_done

bmu2_loop2:
	MOVUPD (R9), X0
	MOVUPD (R10), X1
	MOVUPD X0, X2
	ADDPD  X1, X2
	MULPD  X14, X2
	SUBPD  X2, X0
	MOVUPD X0, (R8)
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  bmu2_loop2

bmu2_done:
	RET

// precisionWeightMulSSE2(dst, errVec, prec []float64)
// ABI0: dst+0(FP)..16, errVec+24(FP)..40, prec+48(FP)..64
TEXT ·precisionWeightMulSSE2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP),        R8
	MOVQ errVec+24(FP),    R9
	MOVQ errVec_len+32(FP), BX
	MOVQ prec+48(FP),      R10

	CMPQ BX, $2
	JL   pw2_done

pw2_loop2:
	MOVUPD (R9), X0
	MOVUPD (R10), X1
	MULPD  X1, X0
	MOVUPD X0, (R8)
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  pw2_loop2

pw2_done:
	RET
