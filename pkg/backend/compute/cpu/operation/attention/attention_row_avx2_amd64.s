#include "textflag.h"

// attentionRowScoresAVX2(scores, q, K []float64, seqLen int, headDim int, scale float64)
//   scores[j] = scale * Σ_d q[d] * K[j*headDim + d]
TEXT ·attentionRowScoresAVX2(SB), NOSPLIT, $0-96
	MOVQ scores+0(FP), AX
	MOVQ q+24(FP), R8
	MOVQ K+48(FP), R9
	MOVQ seqLen+72(FP), R10
	MOVQ headDim+80(FP), R11
	VBROADCASTSD scale+88(FP), Y15

	XORQ R12, R12
ars_j:
	CMPQ R12, R10
	JGE ars_done

	MOVQ R12, BX
	IMULQ R11, BX
	SHLQ $3, BX
	MOVQ R9, SI
	ADDQ BX, SI
	MOVQ R8, DI

	MOVQ R11, CX
	VXORPD Y0, Y0, Y0
	CMPQ CX, $4
	JL ars_dot_tail
ars_dot_loop:
	VMOVUPD (DI), Y1
	VMOVUPD (SI), Y2
	VFMADD231PD Y1, Y2, Y0
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $4, CX
	CMPQ CX, $4
	JGE ars_dot_loop
	VEXTRACTF128 $1, Y0, X3
	VADDPD X3, X0, X0
	VHADDPD X0, X0, X0
ars_dot_tail:
	CMPQ CX, $2
	JL ars_dot_scalar
	MOVUPD (DI), X1
	MOVUPD (SI), X2
	MULPD X2, X1
	HADDPD X1, X1
	ADDSD X1, X0
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, CX
ars_dot_scalar:
	CMPQ CX, $0
	JLE ars_dot_done
	MOVSD (DI), X1
	MOVSD (SI), X2
	MULSD X2, X1
	ADDSD X1, X0
ars_dot_done:
	MULSD X15, X0
	MOVQ R12, BX
	SHLQ $3, BX
	MOVQ AX, SI
	ADDQ BX, SI
	MOVSD X0, (SI)
	INCQ R12
	JMP ars_j

ars_done:
	VZEROUPPER
	RET

TEXT ·attentionRowScoresSSE2(SB), NOSPLIT, $0-96
	MOVQ scores+0(FP), AX
	MOVQ q+24(FP), R8
	MOVQ K+48(FP), R9
	MOVQ seqLen+72(FP), R10
	MOVQ headDim+80(FP), R11
	MOVSD scale+88(FP), X15
	SHUFPD $0, X15, X15

	XORQ R12, R12
arss_j:
	CMPQ R12, R10
	JGE arss_done
	MOVQ R12, BX
	IMULQ R11, BX
	SHLQ $3, BX
	MOVQ R9, SI
	ADDQ BX, SI
	MOVQ R8, DI
	MOVQ R11, CX
	XORPD X0, X0
	CMPQ CX, $2
	JL arss_dot_tail
arss_dot_loop:
	MOVUPD (DI), X1
	MOVUPD (SI), X2
	MULPD X2, X1
	ADDPD X1, X0
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, CX
	CMPQ CX, $2
	JGE arss_dot_loop
	HADDPD X0, X0
arss_dot_tail:
	CMPQ CX, $0
	JLE arss_dot_done
	MOVSD (DI), X1
	MOVSD (SI), X2
	MULSD X2, X1
	ADDSD X1, X0
arss_dot_done:
	MULSD X15, X0
	MOVQ R12, BX
	SHLQ $3, BX
	MOVQ AX, SI
	ADDQ BX, SI
	MOVSD X0, (SI)
	INCQ R12
	JMP arss_j
arss_done:
	RET

// attentionRowOutputAVX2(out, scores, V []float64, seqLen int, headDim int)
//   out[d] = Σ_j scores[j] * V[j*headDim + d]
TEXT ·attentionRowOutputAVX2(SB), NOSPLIT, $0-88
	MOVQ out+0(FP), AX
	MOVQ scores+24(FP), R8
	MOVQ V+48(FP), R9
	MOVQ seqLen+72(FP), R10
	MOVQ headDim+80(FP), R11

	MOVQ R11, CX
	MOVQ AX, SI
	VXORPD Y0, Y0, Y0
aro_clear:
	CMPQ CX, $4
	JL aro_clear_tail
	VMOVUPD Y0, (SI)
	ADDQ $32, SI
	SUBQ $4, CX
	JMP aro_clear
aro_clear_tail:
	CMPQ CX, $2
	JL aro_clear_scalar
	MOVUPD X0, (SI)
	ADDQ $16, SI
	SUBQ $2, CX
aro_clear_scalar:
	CMPQ CX, $0
	JLE aro_clear_done
	MOVSD X0, (SI)
aro_clear_done:

	XORQ R12, R12
aro_j:
	CMPQ R12, R10
	JGE aro_done
	MOVQ R12, BX
	IMULQ R11, BX
	SHLQ $3, BX
	MOVQ R9, SI
	ADDQ BX, SI
	MOVQ AX, DI

	MOVQ R12, BX
	SHLQ $3, BX
	MOVQ R8, DX
	ADDQ BX, DX
	VBROADCASTSD (DX), Y14

	MOVQ R11, CX
	CMPQ CX, $4
	JL aro_w_tail
aro_w_loop:
	VMOVUPD (DI), Y0
	VMOVUPD (SI), Y1
	VFMADD231PD Y14, Y1, Y0
	VMOVUPD Y0, (DI)
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $4, CX
	CMPQ CX, $4
	JGE aro_w_loop
aro_w_tail:
	CMPQ CX, $2
	JL aro_w_scalar
	MOVUPD (DI), X0
	MOVUPD (SI), X1
	MOVAPD X14, X2
	MULPD X1, X2
	ADDPD X2, X0
	MOVUPD X0, (DI)
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, CX
aro_w_scalar:
	CMPQ CX, $0
	JLE aro_w_done
	MOVSD (DI), X0
	MOVSD (SI), X1
	MULSD X14, X1
	ADDSD X1, X0
	MOVSD X0, (DI)
aro_w_done:
	INCQ R12
	JMP aro_j

aro_done:
	VZEROUPPER
	RET

TEXT ·attentionRowOutputSSE2(SB), NOSPLIT, $0-88
	MOVQ out+0(FP), AX
	MOVQ scores+24(FP), R8
	MOVQ V+48(FP), R9
	MOVQ seqLen+72(FP), R10
	MOVQ headDim+80(FP), R11

	MOVQ R11, CX
	MOVQ AX, SI
	XORPD X0, X0
aros_clear:
	CMPQ CX, $2
	JL aros_clear_scalar
	MOVUPD X0, (SI)
	ADDQ $16, SI
	SUBQ $2, CX
	JMP aros_clear
aros_clear_scalar:
	CMPQ CX, $0
	JLE aros_clear_done
	MOVSD X0, (SI)
aros_clear_done:

	XORQ R12, R12
aros_j:
	CMPQ R12, R10
	JGE aros_done
	MOVQ R12, BX
	IMULQ R11, BX
	SHLQ $3, BX
	MOVQ R9, SI
	ADDQ BX, SI
	MOVQ AX, DI
	MOVQ R12, BX
	SHLQ $3, BX
	MOVQ R8, DX
	ADDQ BX, DX
	MOVSD (DX), X14
	SHUFPD $0, X14, X14
	MOVQ R11, CX
	CMPQ CX, $2
	JL aros_w_scalar
aros_w_loop:
	MOVUPD (DI), X0
	MOVUPD (SI), X1
	MOVAPD X14, X2
	MULPD X1, X2
	ADDPD X2, X0
	MOVUPD X0, (DI)
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, CX
	CMPQ CX, $2
	JGE aros_w_loop
aros_w_scalar:
	CMPQ CX, $0
	JLE aros_w_done
	MOVSD (DI), X0
	MOVSD (SI), X1
	MULSD X14, X1
	ADDSD X1, X0
	MOVSD X0, (DI)
aros_w_done:
	INCQ R12
	JMP aros_j
aros_done:
	RET
