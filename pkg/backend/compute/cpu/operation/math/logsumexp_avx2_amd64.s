#include "textflag.h"

// logSumExpRowAVX2(row []float64) float64
// LSE = log(Σ exp(row[i] - max)) + max
TEXT ·logSumExpRowAVX2(SB), NOSPLIT, $0-32
	MOVQ row+0(FP), AX
	MOVQ row_len+8(FP), CX
	CMPQ CX, $0
	JLE lse_done_zero
	MOVQ CX, R11
	MOVQ AX, R12

	// Pass 1: max
	MOVSD (AX), X0
	VBROADCASTSD X0, Y10
	MOVQ AX, R14
	MOVQ CX, R13
	CMPQ R13, $4
	JL lse_max_tail
lse_max_loop:
	VMOVUPD (R14), Y1
	VMAXPD Y1, Y10, Y10
	ADDQ $32, R14
	SUBQ $4, R13
	CMPQ R13, $4
	JGE lse_max_loop
	VEXTRACTF128 $1, Y10, X11
	VMAXPD X11, X10, X10
	MOVAPD X10, X11
	SHUFPD $1, X11, X11
	MAXSD X11, X10
lse_max_tail:
	CMPQ R13, $2
	JL lse_max_scalar
	MOVUPD (R14), X1
	MOVAPD X1, X11
	SHUFPD $1, X11, X11
	MAXSD X11, X1
	MAXSD X1, X10
	ADDQ $16, R14
	SUBQ $2, R13
lse_max_scalar:
	CMPQ R13, $0
	JLE lse_have_max
	MOVSD (R14), X1
	MAXSD X1, X10
lse_have_max:
	VBROADCASTSD X10, Y10

	// Pass 2: sumExp = Σ exp(row[i] - max)
	VBROADCASTSD ·expLog2E(SB), Y4
	VBROADCASTSD ·expLn2Hi(SB), Y5
	VBROADCASTSD ·expLn2Lo(SB), Y6
	VBROADCASTSD ·expMaxArg(SB), Y7
	VBROADCASTSD ·expMinArg(SB), Y8
	VMOVDQU ·expBias32(SB), X9
	VXORPD Y14, Y14, Y14                       // sum

	MOVQ R11, CX
	MOVQ R12, AX
	CMPQ CX, $4
	JL lse_sum_tail
lse_sum_loop:
	VMOVUPD (AX), Y0
	VSUBPD Y10, Y0, Y0
	VMINPD Y7, Y0, Y0
	VMAXPD Y8, Y0, Y0

	VMULPD Y4, Y0, Y1
	VROUNDPD $0, Y1, Y1
	VFNMADD231PD Y5, Y1, Y0
	VFNMADD231PD Y6, Y1, Y0

	VBROADCASTSD ·expC11(SB), Y2
	VBROADCASTSD ·expC10(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·expC9(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·expC8(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·expC7(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·expC6(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·expC5(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·expC4(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·expC3(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·expC2(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·expC1(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·expC0(SB), Y3
	VFMADD213PD Y3, Y0, Y2

	VCVTPD2DQY Y1, X3
	VPADDD X9, X3, X3
	VPMOVSXDQ X3, Y15
	VPSLLQ $52, Y15, Y15
	VMULPD Y15, Y2, Y2
	VADDPD Y2, Y14, Y14

	ADDQ $32, AX
	SUBQ $4, CX
	CMPQ CX, $4
	JGE lse_sum_loop

	// Reduce Y14
	VEXTRACTF128 $1, Y14, X11
	VADDPD X11, X14, X14
	VHADDPD X14, X14, X14

lse_sum_tail:
	CMPQ CX, $0
	JLE lse_post_sum
lse_sum_scalar_loop:
	MOVSD (AX), X0
	SUBSD X10, X0
	MOVSD ·expMaxArg(SB), X1
	MINSD X1, X0
	MOVSD ·expMinArg(SB), X1
	MAXSD X1, X0

	MOVSD ·expLog2E(SB), X1
	MULSD X0, X1
	VROUNDSD $0, X1, X1, X1

	MOVSD ·expLn2Hi(SB), X2
	MULSD X1, X2
	SUBSD X2, X0
	MOVSD ·expLn2Lo(SB), X2
	MULSD X1, X2
	SUBSD X2, X0

	MOVSD ·expC11(SB), X2
	MULSD X0, X2
	ADDSD ·expC10(SB), X2
	MULSD X0, X2
	ADDSD ·expC9(SB), X2
	MULSD X0, X2
	ADDSD ·expC8(SB), X2
	MULSD X0, X2
	ADDSD ·expC7(SB), X2
	MULSD X0, X2
	ADDSD ·expC6(SB), X2
	MULSD X0, X2
	ADDSD ·expC5(SB), X2
	MULSD X0, X2
	ADDSD ·expC4(SB), X2
	MULSD X0, X2
	ADDSD ·expC3(SB), X2
	MULSD X0, X2
	ADDSD ·expC2(SB), X2
	MULSD X0, X2
	ADDSD ·expC1(SB), X2
	MULSD X0, X2
	ADDSD ·expC0(SB), X2

	CVTTSD2SQ X1, BX
	ADDQ $1023, BX
	SHLQ $52, BX
	MOVQ BX, X3
	MULSD X3, X2
	ADDSD X2, X14
	ADDQ $8, AX
	DECQ CX
	JNZ lse_sum_scalar_loop

lse_post_sum:
	// log(sumExp) + max — use logVec via inline path on length-1 buffer in stack.
	// Build a length-1 slice descriptor on stack and call logVecAVX2.
	// Simpler: compute log directly using bit-cast + atanh polynomial inline.
	// X14 holds sumExp.
	MOVSD ·logMantMask(SB), X0
	MOVQ X14, BX
	MOVQ BX, R8                                // copy of bits
	SHRQ $52, BX
	ANDQ $0x7FF, BX
	SUBQ $1023, BX                             // unbiased exponent (signed)
	CVTSQ2SD BX, X1                            // e as double
	MOVQ R8, R9
	MOVQ $0x000FFFFFFFFFFFFF, R10
	ANDQ R10, R9
	MOVQ $0x3FF0000000000000, R10
	ORQ R10, R9
	MOVQ R9, X2                                // m in [1,2)
	MOVSD ·logSqrt2(SB), X3
	UCOMISD X3, X2
	JBE lse_no_shift
	MOVSD ·logHalf(SB), X4
	MULSD X4, X2
	MOVSD $1.0, X4
	ADDSD X4, X1
lse_no_shift:
	// t = (m-1)/(m+1)
	MOVSD $1.0, X3
	MOVAPD X2, X4
	SUBSD X3, X4                               // m-1
	ADDSD X3, X2                               // m+1
	DIVSD X2, X4                               // t
	MOVAPD X4, X5
	MULSD X5, X5                               // u = t²

	// Horner P(u)
	MOVSD ·logA6(SB), X6
	MULSD X5, X6
	ADDSD ·logA5(SB), X6
	MULSD X5, X6
	ADDSD ·logA4(SB), X6
	MULSD X5, X6
	ADDSD ·logA3(SB), X6
	MULSD X5, X6
	ADDSD ·logA2(SB), X6
	MULSD X5, X6
	ADDSD ·logA1(SB), X6
	MULSD X5, X6
	ADDSD ·logA0(SB), X6

	MULSD X4, X6
	ADDSD X6, X6                               // 2*t*P(u) = log(m)

	MOVSD ·logLn2(SB), X7
	MULSD X7, X1
	ADDSD X1, X6                               // log(sumExp)
	ADDSD X10, X6                              // + max
	MOVSD X6, ret+24(FP)
	VZEROUPPER
	RET

lse_done_zero:
	XORPD X0, X0
	MOVSD X0, ret+24(FP)
	VZEROUPPER
	RET
