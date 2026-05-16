#include "textflag.h"

DATA ·atLog2E+0(SB)/8, $1.4426950408889634
GLOBL ·atLog2E(SB), RODATA, $8
DATA ·atLn2Hi+0(SB)/8, $0.6931471805599453
GLOBL ·atLn2Hi(SB), RODATA, $8
DATA ·atLn2Lo+0(SB)/8, $0.0
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
DATA ·atC12+0(SB)/8, $2.08767569878681e-9
GLOBL ·atC12(SB), RODATA, $8
DATA ·atC13+0(SB)/8, $1.6059043836821613e-10
GLOBL ·atC13(SB), RODATA, $8
DATA ·atC14+0(SB)/8, $1.1470745597729725e-11
GLOBL ·atC14(SB), RODATA, $8
DATA ·atC15+0(SB)/8, $7.647163731819816e-13
GLOBL ·atC15(SB), RODATA, $8
DATA ·atC16+0(SB)/8, $4.779477332387385e-14
GLOBL ·atC16(SB), RODATA, $8
DATA ·atC17+0(SB)/8, $2.8114572543455206e-15
GLOBL ·atC17(SB), RODATA, $8
DATA ·atC18+0(SB)/8, $1.5619206968586225e-16
GLOBL ·atC18(SB), RODATA, $8

// scalarTanhAMD64(dst, src []float64)
// tanh(x) per element via (e^{2x}-1)/(e^{2x}+1); NaN → NaN, ±Inf → ±1.
TEXT ·scalarTanhAMD64(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), DI
	MOVQ src_len+32(FP), CX
	CMPQ CX, $0
	JLE tanh_done
tanh_loop:
	MOVSD (DI), X0
	// NaN check
	UCOMISD X0, X0
	JNP tanh_finite
	MOVSD X0, (AX)
	JMP tanh_next
tanh_finite:
	// ±Inf check via comparison with itself×2 (Inf*2=Inf)
	MOVSD ·atMaxArg(SB), X5
	UCOMISD X5, X0
	JBE tanh_check_neg
	MOVSD ·atC0(SB), X1                       // tanh(+Inf) = +1
	MOVSD X1, (AX)
	JMP tanh_next
tanh_check_neg:
	MOVSD ·atMinArg(SB), X5
	UCOMISD X0, X5
	JBE tanh_compute
	MOVSD ·atC0(SB), X1
	XORPD X2, X2
	SUBSD X1, X2                              // -1
	MOVSD X2, (AX)
	JMP tanh_next
tanh_compute:
	// y = 2x
	ADDSD X0, X0
	// clamp (after doubling, still safe range)
	MOVSD ·atMaxArg(SB), X5
	MINSD X5, X0
	MOVSD ·atMinArg(SB), X5
	MAXSD X5, X0

	MOVSD ·atLog2E(SB), X1
	MULSD X0, X1
	CVTSD2SQ X1, BX
	CVTSQ2SD BX, X1

	MOVSD ·atLn2Hi(SB), X2
	MULSD X1, X2
	SUBSD X2, X0
	MOVSD ·atLn2Lo(SB), X2
	MULSD X1, X2
	SUBSD X2, X0

	MOVSD ·atC11(SB), X2
	MULSD X0, X2
	ADDSD ·atC10(SB), X2
	MULSD X0, X2
	ADDSD ·atC9(SB), X2
	MULSD X0, X2
	ADDSD ·atC8(SB), X2
	MULSD X0, X2
	ADDSD ·atC7(SB), X2
	MULSD X0, X2
	ADDSD ·atC6(SB), X2
	MULSD X0, X2
	ADDSD ·atC5(SB), X2
	MULSD X0, X2
	ADDSD ·atC4(SB), X2
	MULSD X0, X2
	ADDSD ·atC3(SB), X2
	MULSD X0, X2
	ADDSD ·atC2(SB), X2
	MULSD X0, X2
	ADDSD ·atC1(SB), X2
	MULSD X0, X2
	ADDSD ·atC0(SB), X2

	ADDQ $1023, BX
	SHLQ $52, BX
	MOVQ BX, X3
	MULSD X3, X2                              // e^{2x}

	MOVSD ·atC0(SB), X4                       // 1.0
	MOVAPD X2, X5
	SUBSD X4, X5                              // e^{2x} - 1
	ADDSD X4, X2                              // e^{2x} + 1
	DIVSD X2, X5
	MOVSD X5, (AX)
tanh_next:
	ADDQ $8, AX
	ADDQ $8, DI
	DECQ CX
	JNZ tanh_loop
tanh_done:
	RET

// scalarSigmoidAMD64(dst, src []float64)
// 1/(1+exp(-x)); NaN → NaN
TEXT ·scalarSigmoidAMD64(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), DI
	MOVQ src_len+32(FP), CX
	CMPQ CX, $0
	JLE sig_done
sig_loop:
	MOVSD (DI), X0
	UCOMISD X0, X0
	JNP sig_finite
	MOVSD X0, (AX)
	JMP sig_next
sig_finite:
	// negate x
	XORPD X4, X4
	SUBSD X0, X4
	MOVAPD X4, X0
	MOVSD ·atMaxArg(SB), X5
	MINSD X5, X0
	MOVSD ·atMinArg(SB), X5
	MAXSD X5, X0

	MOVSD ·atLog2E(SB), X1
	MULSD X0, X1
	CVTSD2SQ X1, BX
	CVTSQ2SD BX, X1

	MOVSD ·atLn2Hi(SB), X2
	MULSD X1, X2
	SUBSD X2, X0
	MOVSD ·atLn2Lo(SB), X2
	MULSD X1, X2
	SUBSD X2, X0

	MOVSD ·atC11(SB), X2
	MULSD X0, X2
	ADDSD ·atC10(SB), X2
	MULSD X0, X2
	ADDSD ·atC9(SB), X2
	MULSD X0, X2
	ADDSD ·atC8(SB), X2
	MULSD X0, X2
	ADDSD ·atC7(SB), X2
	MULSD X0, X2
	ADDSD ·atC6(SB), X2
	MULSD X0, X2
	ADDSD ·atC5(SB), X2
	MULSD X0, X2
	ADDSD ·atC4(SB), X2
	MULSD X0, X2
	ADDSD ·atC3(SB), X2
	MULSD X0, X2
	ADDSD ·atC2(SB), X2
	MULSD X0, X2
	ADDSD ·atC1(SB), X2
	MULSD X0, X2
	ADDSD ·atC0(SB), X2

	ADDQ $1023, BX
	SHLQ $52, BX
	MOVQ BX, X3
	MULSD X3, X2                              // e^{-x}

	MOVSD ·atC0(SB), X4
	ADDSD X4, X2                              // 1 + e^{-x}
	DIVSD X2, X4                              // 1/(1+e^{-x})
	MOVSD X4, (AX)
sig_next:
	ADDQ $8, AX
	ADDQ $8, DI
	DECQ CX
	JNZ sig_loop
sig_done:
	RET

// scalarReLUAMD64(dst, src []float64)
TEXT ·scalarReLUAMD64(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), DI
	MOVQ src_len+32(FP), CX
	XORPD X1, X1
	CMPQ CX, $0
	JLE relu_done
relu_loop:
	MOVSD (DI), X0
	MAXSD X1, X0
	MOVSD X0, (AX)
	ADDQ $8, AX
	ADDQ $8, DI
	DECQ CX
	JNZ relu_loop
relu_done:
	RET

// scalarLeakyReLUAMD64(dst, src []float64, alpha float64)
TEXT ·scalarLeakyReLUAMD64(SB), NOSPLIT, $0-56
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), DI
	MOVQ src_len+32(FP), CX
	MOVSD alpha+48(FP), X8
	XORPD X9, X9
	CMPQ CX, $0
	JLE lrelu_done
lrelu_loop:
	MOVSD (DI), X0
	UCOMISD X9, X0
	JAE lrelu_pos
	MULSD X8, X0
lrelu_pos:
	MOVSD X0, (AX)
	ADDQ $8, AX
	ADDQ $8, DI
	DECQ CX
	JNZ lrelu_loop
lrelu_done:
	RET

// scalarGeLUAMD64(dst, src []float64)
// GeLU(x) = 0.5 x (1 + tanh(sqrt(2/π) (x + 0.044715 x³)))
TEXT ·scalarGeLUAMD64(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), DI
	MOVQ src_len+32(FP), CX
	CMPQ CX, $0
	JLE gelu_done
gelu_loop:
	MOVSD (DI), X0
	UCOMISD X0, X0
	JNP gelu_finite
	MOVSD X0, (AX)
	JMP gelu_next
gelu_finite:
	// z = 0.7978845608028654 * (x + 0.044715 * x³)
	MOVAPD X0, X1
	MULSD X1, X1                              // x²
	MULSD X0, X1                              // x³
	MOVSD ·geluC044+0(SB), X2
	MULSD X2, X1
	ADDSD X0, X1
	MOVSD ·geluC079+0(SB), X2
	MULSD X2, X1                              // z
	// y = 2*z so tanh(z) = (e^{y}-1)/(e^{y}+1)
	ADDSD X1, X1
	// clamp + exp polynomial
	MOVSD ·atMaxArg(SB), X5
	MINSD X5, X1
	MOVSD ·atMinArg(SB), X5
	MAXSD X5, X1

	MOVSD ·atLog2E(SB), X2
	MULSD X1, X2
	CVTSD2SQ X2, BX
	CVTSQ2SD BX, X2

	MOVSD ·atLn2Hi(SB), X3
	MULSD X2, X3
	SUBSD X3, X1
	MOVSD ·atLn2Lo(SB), X3
	MULSD X2, X3
	SUBSD X3, X1

	MOVSD ·atC11(SB), X3
	MULSD X1, X3
	ADDSD ·atC10(SB), X3
	MULSD X1, X3
	ADDSD ·atC9(SB), X3
	MULSD X1, X3
	ADDSD ·atC8(SB), X3
	MULSD X1, X3
	ADDSD ·atC7(SB), X3
	MULSD X1, X3
	ADDSD ·atC6(SB), X3
	MULSD X1, X3
	ADDSD ·atC5(SB), X3
	MULSD X1, X3
	ADDSD ·atC4(SB), X3
	MULSD X1, X3
	ADDSD ·atC3(SB), X3
	MULSD X1, X3
	ADDSD ·atC2(SB), X3
	MULSD X1, X3
	ADDSD ·atC1(SB), X3
	MULSD X1, X3
	ADDSD ·atC0(SB), X3

	ADDQ $1023, BX
	SHLQ $52, BX
	MOVQ BX, X4
	MULSD X4, X3                              // e^{2z}

	MOVSD ·atC0(SB), X4
	MOVAPD X3, X5
	SUBSD X4, X5                              // e^{2z} - 1
	ADDSD X4, X3                              // e^{2z} + 1
	DIVSD X3, X5                              // tanh(z)
	ADDSD X4, X5                              // 1 + tanh(z)
	MULSD X0, X5                              // x*(1+tanh)
	MOVSD ·atC2(SB), X4                       // 0.5
	MULSD X4, X5
	MOVSD X5, (AX)
gelu_next:
	ADDQ $8, AX
	ADDQ $8, DI
	DECQ CX
	JNZ gelu_loop
gelu_done:
	RET

DATA ·geluC044+0(SB)/8, $0.044715
GLOBL ·geluC044(SB), RODATA, $8
DATA ·geluC079+0(SB)/8, $0.7978845608028654
GLOBL ·geluC079(SB), RODATA, $8

// scalarSwiGLUAMD64(dst, src []float64) — gates first half, values second half
TEXT ·scalarSwiGLUAMD64(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ dst_len+8(FP), CX
	MOVQ src+24(FP), DI
	CMPQ CX, $0
	JLE swig_done
	// values pointer = src + half*8
	MOVQ CX, DX
	SHLQ $3, DX
	MOVQ DI, SI
	ADDQ DX, SI                               // SI = values
swig_loop:
	MOVSD (DI), X0                            // gate
	MOVSD (SI), X10                           // value
	UCOMISD X0, X0
	JNP swig_finite
	MOVSD X0, (AX)
	JMP swig_next
swig_finite:
	MOVAPD X0, X11                            // preserve original gate
	XORPD X4, X4
	SUBSD X0, X4
	MOVAPD X4, X0
	MOVSD ·atMaxArg(SB), X5
	MINSD X5, X0
	MOVSD ·atMinArg(SB), X5
	MAXSD X5, X0

	MOVSD ·atLog2E(SB), X1
	MULSD X0, X1
	CVTSD2SQ X1, BX
	CVTSQ2SD BX, X1

	MOVSD ·atLn2Hi(SB), X2
	MULSD X1, X2
	SUBSD X2, X0
	MOVSD ·atLn2Lo(SB), X2
	MULSD X1, X2
	SUBSD X2, X0

	MOVSD ·atC11(SB), X2
	MULSD X0, X2
	ADDSD ·atC10(SB), X2
	MULSD X0, X2
	ADDSD ·atC9(SB), X2
	MULSD X0, X2
	ADDSD ·atC8(SB), X2
	MULSD X0, X2
	ADDSD ·atC7(SB), X2
	MULSD X0, X2
	ADDSD ·atC6(SB), X2
	MULSD X0, X2
	ADDSD ·atC5(SB), X2
	MULSD X0, X2
	ADDSD ·atC4(SB), X2
	MULSD X0, X2
	ADDSD ·atC3(SB), X2
	MULSD X0, X2
	ADDSD ·atC2(SB), X2
	MULSD X0, X2
	ADDSD ·atC1(SB), X2
	MULSD X0, X2
	ADDSD ·atC0(SB), X2

	ADDQ $1023, BX
	SHLQ $52, BX
	MOVQ BX, X3
	MULSD X3, X2                              // e^{-gate}
	MOVSD ·atC0(SB), X4
	ADDSD X4, X2                              // 1+e^{-gate}
	DIVSD X2, X4                              // sigmoid(gate)
	MULSD X11, X4                             // swish(gate) = gate * sigmoid(gate)
	MULSD X10, X4                             // swish(gate) * value
	MOVSD X4, (AX)
swig_next:
	ADDQ $8, AX
	ADDQ $8, DI
	ADDQ $8, SI
	DECQ CX
	JNZ swig_loop
swig_done:
	RET
