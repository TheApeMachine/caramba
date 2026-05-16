#include "textflag.h"
#include "neon_math_arm64.h"

// SigmoidNEON(dst, src []float64)
// Two-lane NEON sigmoid(x) = 1/(1+exp(-x)).
TEXT ·SigmoidNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src_base+24(FP), R1
	MOVD src_len+32(FP), R2
	LSR  $1, R2, R2
	CBZ  R2, sigmoid_neon_done

	VLOADDUP(·atLog2E(SB), R9, V20)
	VLOADDUP(·atLn2Hi(SB), R9, V21)
	VLOADDUP(·atLn2Lo(SB), R9, V22)
	VLOADDUP(·atMaxArg(SB), R9, V23)
	VLOADDUP(·atMinArg(SB), R9, V24)
	VLOADDUP(·atC0(SB), R9, V25)
	MOVD $1023, R10
	VDUP R10, V27.D2
	VEOR V4.B16, V4.B16, V4.B16

sigmoid_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VFSUB_D2(0, 4, 0)
	VFMINNM_D2(23, 0, 0)
	VFMAXNM_D2(24, 0, 0)

	VFMUL_D2(20, 0, 1)
	VFRINTN_D2(1, 1)
	VFMUL_D2(21, 1, 3)
	VFSUB_D2(3, 0, 0)
	VFMUL_D2(22, 1, 3)
	VFSUB_D2(3, 0, 0)

	VLOADDUP(·atC18(SB), R9, V2)
	VLOADDUP(·atC17(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·atC16(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·atC15(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·atC14(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·atC13(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·atC12(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·atC11(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·atC10(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·atC9(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·atC8(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·atC7(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·atC6(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·atC5(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·atC4(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·atC3(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·atC2(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VLOADDUP(·atC1(SB), R9, V3)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(3, 2, 2)
	VFMUL_D2(0, 2, 2)
	VFADD_D2(25, 2, 2)

	VFCVTZS_D2(1, 1)
	VADD V27.D2, V1.D2, V1.D2
	VSHL $52, V1.D2, V1.D2
	VFMUL_D2(1, 2, 2)

	VFADD_D2(25, 2, 2)
	VFDIV_D2(2, 25, 0)
	VST1.P [V0.D2], 16(R0)

	SUBS $1, R2, R2
	BNE  sigmoid_neon_loop

sigmoid_neon_done:
	RET
