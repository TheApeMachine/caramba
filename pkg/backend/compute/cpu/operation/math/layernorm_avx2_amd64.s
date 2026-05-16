#include "textflag.h"

// layerNormRowAVX2(out, row, weight, bias []float64, eps float64)
// Single LayerNorm row, fully fused:
//   mean    = (Σ x) / d
//   var     = (Σ (x-mean)²) / d
//   invStd  = 1 / sqrt(var + eps)
//   out[i]  = (x[i]-mean)*invStd*weight[i] + bias[i]
TEXT ·layerNormRowAVX2(SB), NOSPLIT, $0-104
	MOVQ out+0(FP), AX
	MOVQ row+24(FP), R8
	MOVQ weight+48(FP), R9
	MOVQ bias+72(FP), R10
	MOVQ out_len+8(FP), CX

	MOVQ CX, R11                              // d
	MOVQ CX, R12                              // saved for second pass
	MOVQ R8, R13                              // saved row ptr
	MOVQ R8, R14                              // working row ptr (pass 1)
	VXORPD Y0, Y0, Y0                          // sum accumulator

	CMPQ CX, $4
	JL ln_sum_tail
ln_sum_loop:
	VMOVUPD (R14), Y1
	VADDPD Y1, Y0, Y0
	ADDQ $32, R14
	SUBQ $4, CX
	CMPQ CX, $4
	JGE ln_sum_loop

	VEXTRACTF128 $1, Y0, X2
	VADDPD X2, X0, X0
	VHADDPD X0, X0, X0

ln_sum_tail:
	CMPQ CX, $2
	JL ln_sum_scalar
	MOVUPD (R14), X1
	HADDPD X1, X1
	ADDSD X1, X0
	ADDQ $16, R14
	SUBQ $2, CX
ln_sum_scalar:
	CMPQ CX, $0
	JLE ln_have_sum
	MOVSD (R14), X1
	ADDSD X1, X0
ln_have_sum:
	// mean = X0 / d
	CVTSQ2SD R11, X1
	DIVSD X1, X0
	VBROADCASTSD X0, Y3                       // Y3 = mean broadcast
	MOVSD X0, X3                              // X3 keeps mean

	// pass 2: variance
	MOVQ R13, R14
	MOVQ R12, CX
	VXORPD Y4, Y4, Y4                          // var accumulator
	CMPQ CX, $4
	JL ln_var_tail
ln_var_loop:
	VMOVUPD (R14), Y1
	VSUBPD Y3, Y1, Y1                          // x - mean
	VFMADD231PD Y1, Y1, Y4                     // var += (x-mean)²
	ADDQ $32, R14
	SUBQ $4, CX
	CMPQ CX, $4
	JGE ln_var_loop

	VEXTRACTF128 $1, Y4, X5
	VADDPD X5, X4, X4
	VHADDPD X4, X4, X4

ln_var_tail:
	CMPQ CX, $2
	JL ln_var_scalar
	MOVUPD (R14), X1
	MOVAPD X3, X5
	SHUFPD $0, X5, X5
	SUBPD X5, X1
	MULPD X1, X1
	HADDPD X1, X1
	ADDSD X1, X4
	ADDQ $16, R14
	SUBQ $2, CX
ln_var_scalar:
	CMPQ CX, $0
	JLE ln_have_var
	MOVSD (R14), X1
	SUBSD X3, X1
	MULSD X1, X1
	ADDSD X1, X4
ln_have_var:
	// invStd = 1/sqrt(var/d + eps)
	CVTSQ2SD R11, X5
	DIVSD X5, X4
	ADDSD eps+96(FP), X4
	SQRTSD X4, X4
	MOVSD $1.0, X6
	DIVSD X4, X6
	VBROADCASTSD X6, Y6
	MOVSD X6, X7

	// pass 3: out[i] = (x-mean)*invStd*weight[i] + bias[i]
	MOVQ R13, R14
	MOVQ R12, CX
	CMPQ CX, $4
	JL ln_norm_tail
ln_norm_loop:
	VMOVUPD (R14), Y1
	VSUBPD Y3, Y1, Y1
	VMULPD Y6, Y1, Y1
	VMOVUPD (R9), Y2
	VMULPD Y2, Y1, Y1
	VMOVUPD (R10), Y2
	VADDPD Y2, Y1, Y1
	VMOVUPD Y1, (AX)
	ADDQ $32, AX
	ADDQ $32, R14
	ADDQ $32, R9
	ADDQ $32, R10
	SUBQ $4, CX
	CMPQ CX, $4
	JGE ln_norm_loop

ln_norm_tail:
	CMPQ CX, $2
	JL ln_norm_scalar
	MOVUPD (R14), X1
	MOVAPD X3, X5
	SHUFPD $0, X5, X5
	SUBPD X5, X1
	MOVAPD X7, X8
	SHUFPD $0, X8, X8
	MULPD X8, X1
	MOVUPD (R9), X2
	MULPD X2, X1
	MOVUPD (R10), X2
	ADDPD X2, X1
	MOVUPD X1, (AX)
	ADDQ $16, AX
	ADDQ $16, R14
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $2, CX
ln_norm_scalar:
	CMPQ CX, $0
	JLE ln_done
	MOVSD (R14), X1
	SUBSD X3, X1
	MULSD X7, X1
	MOVSD (R9), X2
	MULSD X2, X1
	MOVSD (R10), X2
	ADDSD X2, X1
	MOVSD X1, (AX)
ln_done:
	VZEROUPPER
	RET
