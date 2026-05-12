#include "textflag.h"

// scalarSqrtTailKernel(dst, src []float64, from int)
//   for i := from; i < len(src); i++ { dst[i] = sqrt(src[i]) }
TEXT ·scalarSqrtTailKernel(SB), NOSPLIT, $0-56
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), DI
	MOVQ src_len+32(FP), CX
	MOVQ from+48(FP), DX
	CMPQ DX, CX
	JGE sst_done
	MOVQ DX, BX
sst_loop:
	CMPQ BX, CX
	JGE sst_done
	MOVQ BX, R8
	SHLQ $3, R8
	MOVSD (DI)(R8*1), X0
	SQRTSD X0, X0
	MOVSD X0, (AX)(R8*1)
	INCQ BX
	JMP sst_loop
sst_done:
	RET

// scalarExpTailKernel(dst, src []float64, from int)
TEXT ·scalarExpTailKernel(SB), NOSPLIT, $0-56
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), DI
	MOVQ src_len+32(FP), CX
	MOVQ from+48(FP), DX
	CMPQ DX, CX
	JGE set_done
sett_loop:
	CMPQ DX, CX
	JGE set_done

	MOVQ DX, R8
	SHLQ $3, R8
	MOVSD (DI)(R8*1), X0

	MOVSD ·expMaxArg(SB), X1
	MINSD X1, X0
	MOVSD ·expMinArg(SB), X1
	MAXSD X1, X0

	MOVSD ·expLog2E(SB), X1
	MULSD X0, X1
	CVTSD2SQ X1, BX
	CVTSQ2SD BX, X1

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

	ADDQ $1023, BX
	SHLQ $52, BX
	MOVQ BX, X3
	MULSD X3, X2

	MOVSD X2, (AX)(R8*1)
	INCQ DX
	JMP sett_loop
set_done:
	RET

// scalarLogTailKernel(dst, src []float64, from int)
TEXT ·scalarLogTailKernel(SB), NOSPLIT, $0-56
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), DI
	MOVQ src_len+32(FP), CX
	MOVQ from+48(FP), DX
slt_loop:
	CMPQ DX, CX
	JGE slt_done

	MOVQ DX, R8
	SHLQ $3, R8
	MOVQ (DI)(R8*1), R9

	MOVQ R9, R10
	SHRQ $52, R10
	ANDQ $0x7FF, R10
	SUBQ $1023, R10
	CVTSQ2SD R10, X1

	MOVQ $0x000FFFFFFFFFFFFF, R11
	ANDQ R11, R9
	MOVQ $0x3FF0000000000000, R11
	ORQ R11, R9
	MOVQ R9, X2

	MOVSD ·logSqrt2(SB), X3
	UCOMISD X3, X2
	JBE slt_no_shift
	MOVSD ·logHalf(SB), X4
	MULSD X4, X2
	MOVSD $1.0, X4
	ADDSD X4, X1
slt_no_shift:
	MOVSD $1.0, X3
	MOVAPD X2, X4
	SUBSD X3, X4
	ADDSD X3, X2
	DIVSD X2, X4
	MOVAPD X4, X5
	MULSD X5, X5

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
	ADDSD X6, X6

	MOVSD ·logLn2(SB), X7
	MULSD X7, X1
	ADDSD X1, X6
	MOVSD X6, (AX)(R8*1)
	INCQ DX
	JMP slt_loop
slt_done:
	RET
