#include "textflag.h"

DATA ·atLog2E+0(SB)/8, $1.4426950408889634
GLOBL ·atLog2E(SB), RODATA, $8
DATA ·atLn2Hi+0(SB)/8, $0.6931471803691864
GLOBL ·atLn2Hi(SB), RODATA, $8
DATA ·atLn2Lo+0(SB)/8, $1.9082149292705877e-10
GLOBL ·atLn2Lo(SB), RODATA, $8
DATA ·atMaxArg+0(SB)/8, $709.0
GLOBL ·atMaxArg(SB), RODATA, $8
DATA ·atMinArg+0(SB)/8, $-708.0
GLOBL ·atMinArg(SB), RODATA, $8
DATA ·atC0+0(SB)/8, $1.0
GLOBL ·atC0(SB), RODATA, $8
DATA ·atC1+0(SB)/8, $1.0
GLOBL ·atC1(SB), RODATA, $8
DATA ·atC2+0(SB)/8, $0.5
GLOBL ·atC2(SB), RODATA, $8
DATA ·atC3+0(SB)/8, $0.16666666666666666
GLOBL ·atC3(SB), RODATA, $8
DATA ·atC4+0(SB)/8, $0.041666666666666664
GLOBL ·atC4(SB), RODATA, $8
DATA ·atC5+0(SB)/8, $0.008333333333333333
GLOBL ·atC5(SB), RODATA, $8
DATA ·atC6+0(SB)/8, $0.001388888888888889
GLOBL ·atC6(SB), RODATA, $8
DATA ·atC7+0(SB)/8, $0.0001984126984126984
GLOBL ·atC7(SB), RODATA, $8
DATA ·atC8+0(SB)/8, $2.4801587301587302e-5
GLOBL ·atC8(SB), RODATA, $8
DATA ·atC9+0(SB)/8, $2.7557319223985893e-6
GLOBL ·atC9(SB), RODATA, $8
DATA ·atC10+0(SB)/8, $2.7557319223985894e-7
GLOBL ·atC10(SB), RODATA, $8
DATA ·atC11+0(SB)/8, $2.5052108385441718e-8
GLOBL ·atC11(SB), RODATA, $8
DATA ·geluC044+0(SB)/8, $0.044715
GLOBL ·geluC044(SB), RODATA, $8
DATA ·geluC079+0(SB)/8, $0.7978845608028654
GLOBL ·geluC079(SB), RODATA, $8

// Helper: expand(x) computes exp(x) in F2, given F0=x. Uses scratch F1..F5, R5..R6.
// All FP constants F20..F31 are reserved by caller pattern. This is inlined by
// each routine below; no separate TEXT block — Plan-9 ARM64 doesn't fold easily.

// scalarTanhNEON(dst, src []float64)
TEXT ·scalarTanhNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD src_len+32(FP), R2
	CBZ R2, tanh_done

	FMOVD ·atLog2E(SB), F20
	FMOVD ·atLn2Hi(SB), F21
	FMOVD ·atLn2Lo(SB), F22
	FMOVD ·atMaxArg(SB), F23
	FMOVD ·atMinArg(SB), F24
	MOVD $1023, R10

tanh_loop:
	FMOVD (R1), F0
	// NaN check: x != x
	FCMPD F0, F0
	BVS tanh_store
	// ±Inf check via finite-clamp: if x > maxArg → +1
	FCMPD F23, F0
	BLE tanh_neg_check
	FMOVD ·atC0(SB), F0
	B tanh_store
tanh_neg_check:
	FCMPD F24, F0
	BGE tanh_compute
	FMOVD ·atC0(SB), F0
	FNEGD F0, F0
	B tanh_store
tanh_compute:
	// y = 2x
	FADDD F0, F0, F0
	FMINNMD F23, F0, F0
	FMAXNMD F24, F0, F0

	FMULD F20, F0, F1
	FRINTND F1, F1
	FMSUBD F1, F0, F21, F0
	FMSUBD F1, F0, F22, F0

	FMOVD ·atC11(SB), F2
	FMOVD ·atC10(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC9(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC8(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC7(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC6(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC5(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC4(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC3(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC2(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC1(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC0(SB), F3
	FMADDD F2, F3, F0, F2

	FCVTZSD F1, R5
	ADD R10, R5, R5
	LSL $52, R5, R5
	FMOVD R5, F4
	FMULD F4, F2, F2                          // e^{2x}

	FMOVD ·atC0(SB), F4
	FSUBD F4, F2, F5                          // e^{2x}-1
	FADDD F4, F2, F2                          // e^{2x}+1
	FDIVD F2, F5, F0
tanh_store:
	FMOVD F0, (R0)
	ADD $8, R0, R0
	ADD $8, R1, R1
	SUBS $1, R2, R2
	BNE tanh_loop
tanh_done:
	RET

// scalarSigmoidNEON(dst, src []float64)
TEXT ·scalarSigmoidNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD src_len+32(FP), R2
	CBZ R2, sig_done

	FMOVD ·atLog2E(SB), F20
	FMOVD ·atLn2Hi(SB), F21
	FMOVD ·atLn2Lo(SB), F22
	FMOVD ·atMaxArg(SB), F23
	FMOVD ·atMinArg(SB), F24
	MOVD $1023, R10

sig_loop:
	FMOVD (R1), F0
	FCMPD F0, F0
	BVS sig_store
	FNEGD F0, F0
	FMINNMD F23, F0, F0
	FMAXNMD F24, F0, F0

	FMULD F20, F0, F1
	FRINTND F1, F1
	FMSUBD F1, F0, F21, F0
	FMSUBD F1, F0, F22, F0

	FMOVD ·atC11(SB), F2
	FMOVD ·atC10(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC9(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC8(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC7(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC6(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC5(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC4(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC3(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC2(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC1(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC0(SB), F3
	FMADDD F2, F3, F0, F2

	FCVTZSD F1, R5
	ADD R10, R5, R5
	LSL $52, R5, R5
	FMOVD R5, F4
	FMULD F4, F2, F2                          // e^{-x}

	FMOVD ·atC0(SB), F4
	FADDD F4, F2, F2                          // 1+e^{-x}
	FDIVD F2, F4, F0                          // 1/(1+e^{-x})
sig_store:
	FMOVD F0, (R0)
	ADD $8, R0, R0
	ADD $8, R1, R1
	SUBS $1, R2, R2
	BNE sig_loop
sig_done:
	RET

// scalarReLUNEON(dst, src []float64)
TEXT ·scalarReLUNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD src_len+32(FP), R2
	FMOVD $0.0, F1
	CBZ R2, relu_done
relu_loop:
	FMOVD (R1), F0
	FMAXNMD F1, F0, F0
	FMOVD F0, (R0)
	ADD $8, R0, R0
	ADD $8, R1, R1
	SUBS $1, R2, R2
	BNE relu_loop
relu_done:
	RET

// scalarLeakyReLUNEON(dst, src []float64, alpha float64)
TEXT ·scalarLeakyReLUNEON(SB), NOSPLIT, $0-56
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD src_len+32(FP), R2
	FMOVD alpha+48(FP), F20
	FMOVD $0.0, F21
	CBZ R2, lr_done
lr_loop:
	FMOVD (R1), F0
	FCMPD F21, F0
	BGE lr_store
	FMULD F20, F0, F0
lr_store:
	FMOVD F0, (R0)
	ADD $8, R0, R0
	ADD $8, R1, R1
	SUBS $1, R2, R2
	BNE lr_loop
lr_done:
	RET

// scalarGeLUNEON(dst, src []float64)
TEXT ·scalarGeLUNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD src_len+32(FP), R2
	CBZ R2, gelu_done

	FMOVD ·atLog2E(SB), F20
	FMOVD ·atLn2Hi(SB), F21
	FMOVD ·atLn2Lo(SB), F22
	FMOVD ·atMaxArg(SB), F23
	FMOVD ·atMinArg(SB), F24
	FMOVD ·geluC044(SB), F25
	FMOVD ·geluC079(SB), F26
	MOVD $1023, R10

gelu_loop:
	FMOVD (R1), F0
	FCMPD F0, F0
	BVS gelu_store
	// z = 0.7978... * (x + 0.044715 * x³)
	FMULD F0, F0, F6
	FMULD F0, F6, F6                          // x³
	FMOVD F0, F7                              // x
	FMADDD F25, F7, F6, F7                    // x + 0.044715*x³
	FMULD F26, F7, F7                         // *0.7978...
	FADDD F7, F7, F7                          // 2z
	FMINNMD F23, F7, F7
	FMAXNMD F24, F7, F7

	FMULD F20, F7, F1
	FRINTND F1, F1
	FMSUBD F1, F7, F21, F7
	FMSUBD F1, F7, F22, F7

	FMOVD ·atC11(SB), F2
	FMOVD ·atC10(SB), F3
	FMADDD F2, F3, F7, F2
	FMOVD ·atC9(SB), F3
	FMADDD F2, F3, F7, F2
	FMOVD ·atC8(SB), F3
	FMADDD F2, F3, F7, F2
	FMOVD ·atC7(SB), F3
	FMADDD F2, F3, F7, F2
	FMOVD ·atC6(SB), F3
	FMADDD F2, F3, F7, F2
	FMOVD ·atC5(SB), F3
	FMADDD F2, F3, F7, F2
	FMOVD ·atC4(SB), F3
	FMADDD F2, F3, F7, F2
	FMOVD ·atC3(SB), F3
	FMADDD F2, F3, F7, F2
	FMOVD ·atC2(SB), F3
	FMADDD F2, F3, F7, F2
	FMOVD ·atC1(SB), F3
	FMADDD F2, F3, F7, F2
	FMOVD ·atC0(SB), F3
	FMADDD F2, F3, F7, F2

	FCVTZSD F1, R5
	ADD R10, R5, R5
	LSL $52, R5, R5
	FMOVD R5, F4
	FMULD F4, F2, F2                          // e^{2z}

	FMOVD ·atC0(SB), F4
	FSUBD F4, F2, F5                          // e^{2z}-1
	FADDD F4, F2, F2                          // e^{2z}+1
	FDIVD F2, F5, F5                          // tanh(z)
	FADDD F4, F5, F5                          // 1+tanh(z)
	FMULD F0, F5, F5                          // x*(1+tanh)
	FMOVD ·atC2(SB), F4                       // 0.5
	FMULD F4, F5, F0
gelu_store:
	FMOVD F0, (R0)
	ADD $8, R0, R0
	ADD $8, R1, R1
	SUBS $1, R2, R2
	BNE gelu_loop
gelu_done:
	RET

// scalarSwiGLUNEON(dst, src []float64)
TEXT ·scalarSwiGLUNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD dst_len+8(FP), R2
	MOVD src+24(FP), R1
	CBZ R2, sw_done
	MOVD R2, R3
	LSL $3, R3, R3
	MOVD R1, R4
	ADD R3, R4, R4                            // values base

	FMOVD ·atLog2E(SB), F20
	FMOVD ·atLn2Hi(SB), F21
	FMOVD ·atLn2Lo(SB), F22
	FMOVD ·atMaxArg(SB), F23
	FMOVD ·atMinArg(SB), F24
	MOVD $1023, R10

sw_loop:
	FMOVD (R1), F0
	FMOVD (R4), F10
	FCMPD F0, F0
	BVS sw_store
	FNEGD F0, F0
	FMINNMD F23, F0, F0
	FMAXNMD F24, F0, F0

	FMULD F20, F0, F1
	FRINTND F1, F1
	FMSUBD F1, F0, F21, F0
	FMSUBD F1, F0, F22, F0

	FMOVD ·atC11(SB), F2
	FMOVD ·atC10(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC9(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC8(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC7(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC6(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC5(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC4(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC3(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC2(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC1(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·atC0(SB), F3
	FMADDD F2, F3, F0, F2

	FCVTZSD F1, R5
	ADD R10, R5, R5
	LSL $52, R5, R5
	FMOVD R5, F4
	FMULD F4, F2, F2                          // e^{-gate}

	FMOVD ·atC0(SB), F4
	FADDD F4, F2, F2                          // 1+e^{-gate}
	FDIVD F2, F4, F4                          // sigmoid
	FMULD F10, F4, F0                         // * value
sw_store:
	FMOVD F0, (R0)
	ADD $8, R0, R0
	ADD $8, R1, R1
	ADD $8, R4, R4
	SUBS $1, R2, R2
	BNE sw_loop
sw_done:
	RET
