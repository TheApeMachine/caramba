#include "textflag.h"

// freeEnergyNEON(mu, expSigma []float64) float64
// Returns sum(mu[i]^2 + expSigma[i]).
// ABI0: mu+0(FP)..16, expSigma+24(FP)..40, ret+48(FP)
TEXT ·freeEnergyNEON(SB), NOSPLIT, $0-56
	MOVD mu+0(FP),        R0
	MOVD mu_len+8(FP),    R1
	MOVD expSigma+24(FP), R2
	FMOVD $0.0, F0
	LSR  $1, R1, R3
	CBZ  R3, fen_scalar

fen_loop2:
	FMOVD.P 8(R0), F1
	FMOVD.P 8(R0), F2
	FMOVD.P 8(R2), F3
	FMOVD.P 8(R2), F4
	FMADDD  F1, F0, F1, F0    // acc += mu^2
	FMADDD  F2, F0, F2, F0
	FADDD   F3, F0, F0         // acc += expSigma
	FADDD   F4, F0, F0
	SUBS $1, R3, R3
	BNE  fen_loop2

fen_scalar:
	AND  $1, R1, R1
	CBZ  R1, fen_done
	FMOVD.P 8(R0), F1
	FMOVD.P 8(R2), F2
	FMADDD  F1, F0, F1, F0
	FADDD   F2, F0, F0

fen_done:
	FMOVD F0, ret+48(FP)
	RET

// beliefUpdateMuNEON(dst, mu, predErr []float64, lr float64)
// dst[i] = mu[i] - lr*(mu[i]+predErr[i])
// ABI0: dst+0(FP)..16, mu+24(FP)..40, predErr+48(FP)..64, lr+72(FP)
TEXT ·beliefUpdateMuNEON(SB), NOSPLIT, $0-80
	MOVD dst+0(FP),       R0
	MOVD mu+24(FP),       R1
	MOVD mu_len+32(FP),   R2
	MOVD predErr+48(FP),  R3
	FMOVD lr+72(FP), F15
	LSR  $1, R2, R4
	CBZ  R4, bmu_scalar

bmu_loop2:
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R1), F1
	FMOVD.P 8(R3), F2
	FMOVD.P 8(R3), F3
	FADDD F2, F0, F4          // mu + predErr
	FADDD F3, F1, F5
	FMSUBD F15, F0, F4, F4   // dst = mu - lr*(mu+pred_err)  = mu - lr*sum
	FMSUBD F15, F1, F5, F5
	FMOVD.P F4, 8(R0)
	FMOVD.P F5, 8(R0)
	SUBS $1, R4, R4
	BNE  bmu_loop2

bmu_scalar:
	AND  $1, R2, R2
	CBZ  R2, bmu_done
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R3), F2
	FADDD F2, F0, F4
	FMSUBD F15, F0, F4, F4
	FMOVD.P F4, 8(R0)

bmu_done:
	RET

// precisionWeightMulNEON(dst, errVec, prec []float64)
// dst[i] = errVec[i] * prec[i]
// ABI0: dst+0(FP)..16, errVec+24(FP)..40, prec+48(FP)..64
TEXT ·precisionWeightMulNEON(SB), NOSPLIT, $0-72
	MOVD dst+0(FP),          R0
	MOVD errVec+24(FP),      R1
	MOVD errVec_len+32(FP),  R2
	MOVD prec+48(FP),        R3
	LSR  $1, R2, R4
	CBZ  R4, pwn_done

pwn_loop2:
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R1), F1
	FMOVD.P 8(R3), F2
	FMOVD.P 8(R3), F3
	FMULD F2, F0, F0
	FMULD F3, F1, F1
	FMOVD.P F0, 8(R0)
	FMOVD.P F1, 8(R0)
	SUBS $1, R4, R4
	BNE  pwn_loop2

pwn_done:
	RET
