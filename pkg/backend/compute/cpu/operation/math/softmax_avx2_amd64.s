#include "textflag.h"

// softmaxRowAVX2(row []float64)
// Numerically-stable in-place softmax over one row:
//   m       = max(row)
//   row[i] -= m
//   row[i]  = exp(row[i])              (degree-11 polynomial, range-reduced)
//   sum     = Σ row[i]
//   row[i] /= sum
//
// Uses the exp* and bias constants declared in exp_avx2_amd64.s.
TEXT ·softmaxRowAVX2(SB), NOSPLIT, $0-24
	MOVQ row+0(FP), AX
	MOVQ row_len+8(FP), CX
	MOVQ CX, R11                              // save n
	MOVQ AX, R12                              // save base ptr
	CMPQ CX, $0
	JLE softmax_done

	// Pass 1: find max
	MOVSD (AX), X0
	VBROADCASTSD X0, Y0
	MOVQ CX, R13
	MOVQ AX, R14
	CMPQ R13, $4
	JL smx_max_tail
smx_max_loop:
	VMOVUPD (R14), Y1
	VMAXPD Y1, Y0, Y0
	ADDQ $32, R14
	SUBQ $4, R13
	CMPQ R13, $4
	JGE smx_max_loop
	// reduce Y0 → scalar in X10
	VEXTRACTF128 $1, Y0, X2
	VMAXPD X2, X0, X0
	MOVAPD X0, X2
	SHUFPD $1, X2, X2
	MAXSD X2, X0
smx_max_tail:
	CMPQ R13, $2
	JL smx_max_scalar
	MOVUPD (R14), X1
	MOVAPD X1, X2
	SHUFPD $1, X2, X2
	MAXSD X2, X1
	MAXSD X1, X0
	ADDQ $16, R14
	SUBQ $2, R13
smx_max_scalar:
	CMPQ R13, $0
	JLE smx_have_max
	MOVSD (R14), X1
	MAXSD X1, X0
smx_have_max:
	VBROADCASTSD X0, Y10                       // max broadcast
	MOVSD X0, X10                              // scalar max

	// Pass 2: exp(row[i] - max)  in-place, reuse exp constants
	VBROADCASTSD ·expLog2E(SB), Y4
	VBROADCASTSD ·expLn2Hi(SB), Y5
	VBROADCASTSD ·expLn2Lo(SB), Y6
	VBROADCASTSD ·expMaxArg(SB), Y7
	VBROADCASTSD ·expMinArg(SB), Y8
	VMOVDQU ·expBias32(SB), X9

	MOVQ R11, CX
	MOVQ R12, AX
	CMPQ CX, $4
	JL smx_exp_tail
smx_exp_loop:
	VMOVUPD (AX), Y0
	VSUBPD  Y10, Y0, Y0
	VMINPD  Y7, Y0, Y0
	VMAXPD  Y8, Y0, Y0

	VMULPD   Y4, Y0, Y1
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
	VMOVUPD Y2, (AX)

	ADDQ $32, AX
	SUBQ $4, CX
	CMPQ CX, $4
	JGE smx_exp_loop

smx_exp_tail:
	CMPQ CX, $0
	JLE smx_post_exp
smx_exp_scalar_loop:
	// scalar fallback using SQRTSD-style polynomial exp would be too verbose;
	// reuse the existing scalar Go path via go:noescape — but to stay assembly-only,
	// process via single-lane VEX ops on X registers using the same constants.
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
	MOVSD X2, (AX)
	ADDQ $8, AX
	DECQ CX
	JNZ smx_exp_scalar_loop

smx_post_exp:
	// Pass 3: sum
	VXORPD Y11, Y11, Y11
	MOVQ R11, CX
	MOVQ R12, AX
	CMPQ CX, $4
	JL smx_sum_tail
smx_sum_loop:
	VMOVUPD (AX), Y0
	VADDPD Y0, Y11, Y11
	ADDQ $32, AX
	SUBQ $4, CX
	CMPQ CX, $4
	JGE smx_sum_loop
	VEXTRACTF128 $1, Y11, X12
	VADDPD X12, X11, X11
	VHADDPD X11, X11, X11
smx_sum_tail:
	CMPQ CX, $2
	JL smx_sum_scalar
	MOVUPD (AX), X0
	HADDPD X0, X0
	ADDSD X0, X11
	ADDQ $16, AX
	SUBQ $2, CX
smx_sum_scalar:
	CMPQ CX, $0
	JLE smx_have_sum
	MOVSD (AX), X0
	ADDSD X0, X11
smx_have_sum:
	// Pass 4: row[i] /= sum
	MOVSD $1.0, X12
	DIVSD X11, X12
	VBROADCASTSD X12, Y12

	MOVQ R11, CX
	MOVQ R12, AX
	CMPQ CX, $4
	JL smx_div_tail
smx_div_loop:
	VMOVUPD (AX), Y0
	VMULPD Y12, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	SUBQ $4, CX
	CMPQ CX, $4
	JGE smx_div_loop
smx_div_tail:
	CMPQ CX, $2
	JL smx_div_scalar
	MOVUPD (AX), X0
	MOVAPD X12, X1
	SHUFPD $0, X1, X1
	MULPD X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	SUBQ $2, CX
smx_div_scalar:
	CMPQ CX, $0
	JLE softmax_done
	MOVSD (AX), X0
	MULSD X12, X0
	MOVSD X0, (AX)

softmax_done:
	VZEROUPPER
	RET
