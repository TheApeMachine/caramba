#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// freeEnergyNEON(mu, expSigma []float64) float64
// Returns sum(mu[i]^2 + expSigma[i]).
// ABI0: mu+0(FP)..16, expSigma+24(FP)..40, ret+48(FP)
TEXT ·freeEnergyNEON(SB), NOSPLIT, $16-56
	MOVD mu+0(FP),        R0
	MOVD mu_len+8(FP),    R1
	MOVD expSigma+24(FP), R2
	FMOVD $0.0, F0
	LSR  $1, R1, R3
	CBZ  R3, fen_scalar
	VEOR V0.B16, V0.B16, V0.B16

fen_loop2:
	VLD1.P 16(R0), [V1.D2]
	VLD1.P 16(R2), [V2.D2]
	VFMUL_D2(1, 1, 3)
	VFADD_D2(2, 3, 3)
	VFADD_D2(3, 0, 0)
	SUBS $1, R3, R3
	BNE  fen_loop2

	MOVD RSP, R3
	VST1.P [V0.D2], 16(R3)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0

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
TEXT ·beliefUpdateMuNEON(SB), NOSPLIT, $8-80
	MOVD dst+0(FP),       R0
	MOVD mu+24(FP),       R1
	MOVD mu_len+32(FP),   R2
	MOVD predErr+48(FP),  R3
	FMOVD lr+72(FP), F15
	FMOVD F15, 0(RSP)
	VLD1R (RSP), [V15.D2]
	LSR  $1, R2, R4
	CBZ  R4, bmu_scalar

bmu_loop2:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R3), [V2.D2]
	VFADD_D2(2, 0, 4)
	VFMUL_D2(15, 4, 4)
	VFSUB_D2(4, 0, 4)
	VST1.P [V4.D2], 16(R0)
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
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R3), [V1.D2]
	VFMUL_D2(1, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R4, R4
	BNE  pwn_loop2

	TST $1, R2
	BEQ pwn_done
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R3), F2
	FMULD F2, F0, F0
	FMOVD.P F0, 8(R0)

pwn_done:
	RET
