#include "textflag.h"

// scalarSqrtTailKernel(dst, src []float64, from int)
TEXT ·scalarSqrtTailKernel(SB), NOSPLIT, $0-56
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD src_len+32(FP), R2
	MOVD from+48(FP), R3
sst_loop:
	CMP R2, R3
	BGE sst_done
	LSL $3, R3, R4
	ADD R1, R4, R5
	FMOVD (R5), F0
	FSQRTD F0, F0
	ADD R0, R4, R5
	FMOVD F0, (R5)
	ADD $1, R3, R3
	B sst_loop
sst_done:
	RET

// scalarExpTailKernel(dst, src []float64, from int)
TEXT ·scalarExpTailKernel(SB), NOSPLIT, $0-56
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD src_len+32(FP), R2
	MOVD from+48(FP), R3
	FMOVD ·expLog2E(SB), F20
	FMOVD ·expLn2Hi(SB), F21
	FMOVD ·expLn2Lo(SB), F22
	FMOVD ·expMaxArg(SB), F23
	FMOVD ·expMinArg(SB), F24
	MOVD $1023, R10
set_loop:
	CMP R2, R3
	BGE set_done
	LSL $3, R3, R4
	ADD R1, R4, R5
	FMOVD (R5), F0
	FMINNMD F23, F0, F0
	FMAXNMD F24, F0, F0

	FMULD F20, F0, F1
	FRINTND F1, F1
	FMSUBD F1, F0, F21, F0
	FMSUBD F1, F0, F22, F0

	FMOVD ·expC11(SB), F2
	FMOVD ·expC10(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·expC9(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·expC8(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·expC7(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·expC6(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·expC5(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·expC4(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·expC3(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·expC2(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·expC1(SB), F3
	FMADDD F2, F3, F0, F2
	FMOVD ·expC0(SB), F3
	FMADDD F2, F3, F0, F2

	FCVTZSD F1, R5
	ADD R10, R5, R5
	LSL $52, R5, R5
	FMOVD R5, F4
	FMULD F4, F2, F2

	LSL $3, R3, R4
	ADD R0, R4, R5
	FMOVD F2, (R5)
	ADD $1, R3, R3
	B set_loop
set_done:
	RET

// scalarLogTailKernel(dst, src []float64, from int)
TEXT ·scalarLogTailKernel(SB), NOSPLIT, $0-56
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD src_len+32(FP), R2
	MOVD from+48(FP), R3
	MOVD $1023, R10
slt_loop:
	CMP R2, R3
	BGE slt_done
	LSL $3, R3, R4
	ADD R1, R4, R5
	FMOVD (R5), F0
	FMOVD F0, R6
	LSR $52, R6, R7
	AND $0x7FF, R7, R7
	SUB R10, R7, R7
	SCVTFD R7, F8

	MOVD $0x000FFFFFFFFFFFFF, R8
	AND R8, R6, R8
	MOVD $0x3FF0000000000000, R9
	ORR R9, R8, R8
	FMOVD R8, F9

	FMOVD ·logSqrt2(SB), F1
	FCMPD F1, F9
	BLE slt_no_shift
	FMOVD ·logHalf(SB), F2
	FMULD F2, F9, F9
	FMOVD $1.0, F2
	FADDD F2, F8, F8
slt_no_shift:
	FMOVD $1.0, F1
	FSUBD F1, F9, F2
	FADDD F1, F9, F3
	FDIVD F3, F2, F2
	FMULD F2, F2, F3

	FMOVD ·logA6(SB), F4
	FMOVD ·logA5(SB), F5
	FMADDD F4, F5, F3, F4
	FMOVD ·logA4(SB), F5
	FMADDD F4, F5, F3, F4
	FMOVD ·logA3(SB), F5
	FMADDD F4, F5, F3, F4
	FMOVD ·logA2(SB), F5
	FMADDD F4, F5, F3, F4
	FMOVD ·logA1(SB), F5
	FMADDD F4, F5, F3, F4
	FMOVD ·logA0(SB), F5
	FMADDD F4, F5, F3, F4

	FMULD F2, F4, F4
	FADDD F4, F4, F4

	FMOVD ·logLn2(SB), F5
	FMADDD F5, F4, F8, F4

	LSL $3, R3, R4
	ADD R0, R4, R5
	FMOVD F4, (R5)
	ADD $1, R3, R3
	B slt_loop
slt_done:
	RET
