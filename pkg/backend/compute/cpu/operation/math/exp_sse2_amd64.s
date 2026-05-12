#include "textflag.h"

// expVecSSE2(dst, src []float64)
// Same algorithm as expVecAVX2, 2 lanes per iteration, no FMA / no ROUNDPD.
// Rounding is delegated to CVTPD2DQ (round-to-nearest by default) which
// implicitly handles the magic-number-style rounding required for range
// reduction.
TEXT ·expVecSSE2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), DI
	MOVQ src_len+32(FP), BX

	MOVSD  ·expLog2E(SB), X10
	SHUFPD $0, X10, X10
	MOVSD  ·expLn2Hi(SB), X11
	SHUFPD $0, X11, X11
	MOVSD  ·expLn2Lo(SB), X12
	SHUFPD $0, X12, X12
	MOVSD  ·expMaxArg(SB), X13
	SHUFPD $0, X13, X13
	MOVSD  ·expMinArg(SB), X14
	SHUFPD $0, X14, X14
	MOVOU  ·expBias32(SB), X15

	CMPQ BX, $2
	JL   done_exp_sse2
loop_exp_sse2:
	MOVUPD (DI), X0
	MINPD  X13, X0
	MAXPD  X14, X0

	MOVAPD X0, X1
	MULPD  X10, X1

	CVTPD2PL X1, X4
	CVTPL2PD X4, X1

	MOVAPD X1, X5
	MULPD  X11, X5
	SUBPD  X5, X0
	MOVAPD X1, X5
	MULPD  X12, X5
	SUBPD  X5, X0

	MOVSD  ·expC11(SB), X2
	SHUFPD $0, X2, X2
	MULPD  X0, X2
	MOVSD  ·expC10(SB), X3
	SHUFPD $0, X3, X3
	ADDPD  X3, X2

	MULPD  X0, X2
	MOVSD  ·expC9(SB), X3
	SHUFPD $0, X3, X3
	ADDPD  X3, X2

	MULPD  X0, X2
	MOVSD  ·expC8(SB), X3
	SHUFPD $0, X3, X3
	ADDPD  X3, X2

	MULPD  X0, X2
	MOVSD  ·expC7(SB), X3
	SHUFPD $0, X3, X3
	ADDPD  X3, X2

	MULPD  X0, X2
	MOVSD  ·expC6(SB), X3
	SHUFPD $0, X3, X3
	ADDPD  X3, X2

	MULPD  X0, X2
	MOVSD  ·expC5(SB), X3
	SHUFPD $0, X3, X3
	ADDPD  X3, X2

	MULPD  X0, X2
	MOVSD  ·expC4(SB), X3
	SHUFPD $0, X3, X3
	ADDPD  X3, X2

	MULPD  X0, X2
	MOVSD  ·expC3(SB), X3
	SHUFPD $0, X3, X3
	ADDPD  X3, X2

	MULPD  X0, X2
	MOVSD  ·expC2(SB), X3
	SHUFPD $0, X3, X3
	ADDPD  X3, X2

	MULPD  X0, X2
	MOVSD  ·expC1(SB), X3
	SHUFPD $0, X3, X3
	ADDPD  X3, X2

	MULPD  X0, X2
	MOVSD  ·expC0(SB), X3
	SHUFPD $0, X3, X3
	ADDPD  X3, X2

	PADDD     X15, X4
	PXOR      X5, X5
	PUNPCKLLQ X5, X4
	PSLLQ     $52, X4

	MULPD  X4, X2
	MOVUPD X2, (AX)

	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_exp_sse2

done_exp_sse2:
	RET
