#include "textflag.h"

// rmsNormRowAVX2(out, row, weight []float64, eps float64)
//   rms    = sqrt((Σ x²)/d + eps)
//   invRMS = 1/rms
//   out[i] = x[i] * invRMS * weight[i]
TEXT ·rmsNormRowAVX2(SB), NOSPLIT, $0-80
	MOVQ out+0(FP), AX
	MOVQ row+24(FP), R8
	MOVQ weight+48(FP), R9
	MOVQ out_len+8(FP), CX
	MOVQ CX, R11                              // d
	MOVQ CX, R12                              // pass-2 counter
	MOVQ R8, R13                              // saved row ptr
	MOVQ R8, R14
	VXORPD Y0, Y0, Y0

	CMPQ CX, $4
	JL rms_sum_tail
rms_sum_loop:
	VMOVUPD (R14), Y1
	VFMADD231PD Y1, Y1, Y0
	ADDQ $32, R14
	SUBQ $4, CX
	CMPQ CX, $4
	JGE rms_sum_loop
	VEXTRACTF128 $1, Y0, X2
	VADDPD X2, X0, X0
	VHADDPD X0, X0, X0

rms_sum_tail:
	CMPQ CX, $2
	JL rms_sum_scalar
	MOVUPD (R14), X1
	MULPD X1, X1
	HADDPD X1, X1
	ADDSD X1, X0
	ADDQ $16, R14
	SUBQ $2, CX
rms_sum_scalar:
	CMPQ CX, $0
	JLE rms_have_sum
	MOVSD (R14), X1
	MULSD X1, X1
	ADDSD X1, X0
rms_have_sum:
	CVTSQ2SD R11, X1
	DIVSD X1, X0
	ADDSD eps+72(FP), X0
	SQRTSD X0, X0
	MOVSD $1.0, X3
	DIVSD X0, X3
	VBROADCASTSD X3, Y3
	MOVSD X3, X4

	MOVQ R13, R14
	MOVQ R12, CX
	CMPQ CX, $4
	JL rms_norm_tail
rms_norm_loop:
	VMOVUPD (R14), Y1
	VMULPD Y3, Y1, Y1
	VMOVUPD (R9), Y2
	VMULPD Y2, Y1, Y1
	VMOVUPD Y1, (AX)
	ADDQ $32, AX
	ADDQ $32, R14
	ADDQ $32, R9
	SUBQ $4, CX
	CMPQ CX, $4
	JGE rms_norm_loop

rms_norm_tail:
	CMPQ CX, $2
	JL rms_norm_scalar
	MOVUPD (R14), X1
	MOVAPD X4, X5
	SHUFPD $0, X5, X5
	MULPD X5, X1
	MOVUPD (R9), X2
	MULPD X2, X1
	MOVUPD X1, (AX)
	ADDQ $16, AX
	ADDQ $16, R14
	ADDQ $16, R9
	SUBQ $2, CX
rms_norm_scalar:
	CMPQ CX, $0
	JLE rms_done
	MOVSD (R14), X1
	MULSD X4, X1
	MOVSD (R9), X2
	MULSD X2, X1
	MOVSD X1, (AX)
rms_done:
	VZEROUPPER
	RET

TEXT ·rmsNormRowSSE2(SB), NOSPLIT, $0-80
	MOVQ out+0(FP), AX
	MOVQ row+24(FP), R8
	MOVQ weight+48(FP), R9
	MOVQ out_len+8(FP), CX
	MOVQ CX, R11
	MOVQ CX, R12
	MOVQ R8, R13
	MOVQ R8, R14
	XORPD X0, X0

	CMPQ CX, $2
	JL rmss_sum_tail
rmss_sum_loop:
	MOVUPD (R14), X1
	MULPD X1, X1
	ADDPD X1, X0
	ADDQ $16, R14
	SUBQ $2, CX
	CMPQ CX, $2
	JGE rmss_sum_loop
	HADDPD X0, X0

rmss_sum_tail:
	CMPQ CX, $0
	JLE rmss_have_sum
	MOVSD (R14), X1
	MULSD X1, X1
	ADDSD X1, X0
rmss_have_sum:
	CVTSQ2SD R11, X1
	DIVSD X1, X0
	ADDSD eps+72(FP), X0
	SQRTSD X0, X0
	MOVSD $1.0, X3
	DIVSD X0, X3
	MOVAPD X3, X4
	SHUFPD $0, X4, X4

	MOVQ R13, R14
	MOVQ R12, CX
	CMPQ CX, $2
	JL rmss_norm_tail
rmss_norm_loop:
	MOVUPD (R14), X1
	MULPD X4, X1
	MOVUPD (R9), X2
	MULPD X2, X1
	MOVUPD X1, (AX)
	ADDQ $16, AX
	ADDQ $16, R14
	ADDQ $16, R9
	SUBQ $2, CX
	CMPQ CX, $2
	JGE rmss_norm_loop

rmss_norm_tail:
	CMPQ CX, $0
	JLE rmss_done
	MOVSD (R14), X1
	MULSD X3, X1
	MOVSD (R9), X2
	MULSD X2, X1
	MOVSD X1, (AX)
rmss_done:
	RET
