#include "textflag.h"

// hawkesKernelRowAVX2(out, events []float64, ti, alpha, beta float64)
//   out[i] = alpha * exp(-beta * (events[i] - ti))
TEXT ·hawkesKernelRowAVX2(SB), NOSPLIT, $0-72
	MOVQ out+0(FP), AX
	MOVQ events+24(FP), DI
	MOVQ out_len+8(FP), CX
	CMPQ CX, $0
	JLE hkr_done

	VBROADCASTSD ti+48(FP), Y10
	VBROADCASTSD beta+64(FP), Y11
	VXORPD Y15, Y15, Y15
	VSUBPD Y11, Y15, Y11                       // negBeta
	VBROADCASTSD alpha+56(FP), Y12
	VBROADCASTSD ·hexLog2E(SB), Y4
	VBROADCASTSD ·hexLn2Hi(SB), Y5
	VBROADCASTSD ·hexLn2Lo(SB), Y6
	VBROADCASTSD ·hexMaxArg(SB), Y7
	VBROADCASTSD ·hexMinArg(SB), Y8
	VMOVDQU ·hexBias32(SB), X9

	CMPQ CX, $4
	JL hkr_tail
hkr_loop:
	VMOVUPD (DI), Y0
	VSUBPD Y10, Y0, Y0                         // events - ti
	VMULPD Y11, Y0, Y0                         // -beta * (events-ti)
	VMINPD Y7, Y0, Y0
	VMAXPD Y8, Y0, Y0

	VMULPD Y4, Y0, Y1
	VROUNDPD $0, Y1, Y1
	VFNMADD231PD Y5, Y1, Y0
	VFNMADD231PD Y6, Y1, Y0

	// See excitation_avx2_amd64.s for the rationale on not hoisting the
	// 12-coefficient table — register pressure exceeds the 16-YMM budget.
	VBROADCASTSD ·hexC11(SB), Y2
	VBROADCASTSD ·hexC10(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·hexC9(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·hexC8(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·hexC7(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·hexC6(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·hexC5(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·hexC4(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·hexC3(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·hexC2(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·hexC1(SB), Y3
	VFMADD213PD Y3, Y0, Y2
	VBROADCASTSD ·hexC0(SB), Y3
	VFMADD213PD Y3, Y0, Y2

	VCVTPD2DQY Y1, X13
	VPADDD X9, X13, X13
	VPMOVSXDQ X13, Y14
	VPSLLQ $52, Y14, Y14
	VMULPD Y14, Y2, Y2
	VMULPD Y12, Y2, Y2                         // *= alpha
	VMOVUPD Y2, (AX)

	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, CX
	CMPQ CX, $4
	JGE hkr_loop

hkr_tail:
	CMPQ CX, $0
	JLE hkr_done
hkr_scalar:
	MOVSD (DI), X0
	MOVSD ti+48(FP), X1
	SUBSD X1, X0
	MOVSD beta+64(FP), X1
	XORPD X3, X3
	SUBSD X1, X3
	MULSD X3, X0

	MOVSD ·hexMaxArg(SB), X1
	MINSD X1, X0
	MOVSD ·hexMinArg(SB), X1
	MAXSD X1, X0

	MOVSD ·hexLog2E(SB), X1
	MULSD X0, X1
	ROUNDSD $0, X1, X1                         // explicit round-to-nearest-even
	CVTTSD2SQ X1, BX

	MOVSD ·hexLn2Hi(SB), X2
	MULSD X1, X2
	SUBSD X2, X0
	MOVSD ·hexLn2Lo(SB), X2
	MULSD X1, X2
	SUBSD X2, X0

	MOVSD ·hexC11(SB), X2
	MULSD X0, X2
	ADDSD ·hexC10(SB), X2
	MULSD X0, X2
	ADDSD ·hexC9(SB), X2
	MULSD X0, X2
	ADDSD ·hexC8(SB), X2
	MULSD X0, X2
	ADDSD ·hexC7(SB), X2
	MULSD X0, X2
	ADDSD ·hexC6(SB), X2
	MULSD X0, X2
	ADDSD ·hexC5(SB), X2
	MULSD X0, X2
	ADDSD ·hexC4(SB), X2
	MULSD X0, X2
	ADDSD ·hexC3(SB), X2
	MULSD X0, X2
	ADDSD ·hexC2(SB), X2
	MULSD X0, X2
	ADDSD ·hexC1(SB), X2
	MULSD X0, X2
	ADDSD ·hexC0(SB), X2

	ADDQ $1023, BX
	SHLQ $52, BX
	MOVQ BX, X3
	MULSD X3, X2
	MULSD alpha+56(FP), X2
	MOVSD X2, (AX)

	ADDQ $8, AX
	ADDQ $8, DI
	DECQ CX
	JNZ hkr_scalar

hkr_done:
	VZEROUPPER
	RET

TEXT ·hawkesKernelRowSSE2(SB), NOSPLIT, $0-72
	MOVQ out+0(FP), AX
	MOVQ events+24(FP), DI
	MOVQ out_len+8(FP), CX
	CMPQ CX, $0
	JLE hkrs_done
hkrs_loop:
	MOVSD (DI), X0
	MOVSD ti+48(FP), X1
	SUBSD X1, X0
	MOVSD beta+64(FP), X1
	XORPD X3, X3
	SUBSD X1, X3
	MULSD X3, X0

	MOVSD ·hexMaxArg(SB), X1
	MINSD X1, X0
	MOVSD ·hexMinArg(SB), X1
	MAXSD X1, X0

	MOVSD ·hexLog2E(SB), X1
	MULSD X0, X1
	ROUNDSD $0, X1, X1                         // explicit round-to-nearest-even
	CVTTSD2SQ X1, BX

	MOVSD ·hexLn2Hi(SB), X2
	MULSD X1, X2
	SUBSD X2, X0
	MOVSD ·hexLn2Lo(SB), X2
	MULSD X1, X2
	SUBSD X2, X0

	MOVSD ·hexC11(SB), X2
	MULSD X0, X2
	ADDSD ·hexC10(SB), X2
	MULSD X0, X2
	ADDSD ·hexC9(SB), X2
	MULSD X0, X2
	ADDSD ·hexC8(SB), X2
	MULSD X0, X2
	ADDSD ·hexC7(SB), X2
	MULSD X0, X2
	ADDSD ·hexC6(SB), X2
	MULSD X0, X2
	ADDSD ·hexC5(SB), X2
	MULSD X0, X2
	ADDSD ·hexC4(SB), X2
	MULSD X0, X2
	ADDSD ·hexC3(SB), X2
	MULSD X0, X2
	ADDSD ·hexC2(SB), X2
	MULSD X0, X2
	ADDSD ·hexC1(SB), X2
	MULSD X0, X2
	ADDSD ·hexC0(SB), X2

	ADDQ $1023, BX
	SHLQ $52, BX
	MOVQ BX, X3
	MULSD X3, X2
	MULSD alpha+56(FP), X2
	MOVSD X2, (AX)

	ADDQ $8, AX
	ADDQ $8, DI
	DECQ CX
	JNZ hkrs_loop

hkrs_done:
	RET
