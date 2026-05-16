#include "textflag.h"

// freeEnergyAVX2(mu, expSigma []float64) float64
// Computes sum(mu[i]^2 + expSigma[i]) over all i.
// Caller subtracts logSigma+1 terms and multiplies by 0.5.
// ABI0: mu+0(FP)..16, expSigma+24(FP)..40, ret+48(FP)
TEXT ·freeEnergyAVX2(SB), NOSPLIT, $0-56
	MOVQ mu+0(FP),        AX
	MOVQ mu_len+8(FP),    BX
	MOVQ expSigma+24(FP), DI
	VXORPD Y0, Y0, Y0          // acc = 0

	CMPQ BX, $4
	JL   fe_tail

fe_loop4:
	VMOVUPD (AX), Y1            // mu[i..i+3]
	VMOVUPD (DI), Y2            // expSigma[i..i+3]
	VFMADD231PD Y1, Y1, Y0     // acc += mu^2
	VADDPD Y2, Y0, Y0           // acc += expSigma
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  fe_loop4

fe_tail:
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0

	TESTQ BX, BX
	JZ    fe_done
fe_scalar:
	VMOVSD (AX), X1
	VMOVSD (DI), X2
	VFMADD231SD X1, X1, X0
	VADDSD X2, X0, X0
	ADDQ $8, AX
	ADDQ $8, DI
	DECQ BX
	JNZ  fe_scalar

fe_done:
	MOVSD X0, ret+48(FP)
	VZEROUPPER
	RET

// beliefUpdateMuAVX2(dst, mu, predErr []float64, lr float64)
// dst[i] = mu[i] - lr*(mu[i]+predErr[i])  = mu[i]*(1-lr) - lr*predErr[i]
// ABI0: dst+0(FP)..16, mu+24(FP)..40, predErr+48(FP)..64, lr+72(FP)
TEXT ·beliefUpdateMuAVX2(SB), NOSPLIT, $0-80
	MOVQ dst+0(FP),      R8
	MOVQ mu+24(FP),      R9
	MOVQ mu_len+32(FP),  BX
	MOVQ predErr+48(FP), R10
	VMOVSD lr+72(FP), X14
	VBROADCASTSD X14, Y14             // Y14 = lr (broadcast)
	// Build 1.0 in X15 via integer constant 0x3FF0000000000000
	MOVQ $0x3FF0000000000000, AX
	MOVQ AX, X15
	VBROADCASTSD X15, Y15
	VSUBPD Y14, Y15, Y15              // Y15 = (1-lr)

	CMPQ BX, $4
	JL   bmu_tail

bmu_loop4:
	VMOVUPD (R9), Y0                  // mu
	VMOVUPD (R10), Y1                 // predErr
	VMULPD  Y15, Y0, Y2               // mu*(1-lr)
	VFNMADD231PD Y14, Y1, Y2         // Y2 -= lr*predErr
	VMOVUPD Y2, (R8)
	ADDQ $32, R8
	ADDQ $32, R9
	ADDQ $32, R10
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  bmu_loop4

bmu_tail:
	VZEROUPPER
	RET

// precisionWeightMulAVX2(dst, errVec, prec []float64)
// dst[i] = errVec[i] * prec[i]
// ABI0: dst+0(FP)..16, errVec+24(FP)..40, prec+48(FP)..64
TEXT ·precisionWeightMulAVX2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP),        R8
	MOVQ errVec+24(FP),    R9
	MOVQ errVec_len+32(FP), BX
	MOVQ prec+48(FP),      R10

	CMPQ BX, $4
	JL   pw_done

pw_loop4:
	VMOVUPD (R9), Y0
	VMOVUPD (R10), Y1
	VMULPD  Y1, Y0, Y0
	VMOVUPD Y0, (R8)
	ADDQ $32, R8
	ADDQ $32, R9
	ADDQ $32, R10
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  pw_loop4

pw_done:
	VZEROUPPER
	RET
