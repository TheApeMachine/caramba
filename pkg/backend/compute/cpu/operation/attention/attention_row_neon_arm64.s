#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// attentionRowScoresNEON computes the per-row attention score vector:
//   scores[j] = scale * Σ_d q[d] * K[j*headDim + d]   for j in 0..seqLen
//
// Guards: returns immediately for seqLen=0; per-row, returns 0 for headDim=0.
//
// ABI0 layout:
//   scores+0(FP)..23, q+24(FP)..47, K+48(FP)..71,
//   seqLen+72(FP), headDim+80(FP), scale+88(FP)
TEXT ·attentionRowScoresNEON(SB), NOSPLIT, $16-96
	MOVD scores+0(FP), R0
	MOVD q+24(FP), R1
	MOVD K+48(FP), R2
	MOVD seqLen+72(FP), R3
	MOVD headDim+80(FP), R4
	FMOVD scale+88(FP), F20

	CBZ R3, ars_done                          // seqLen == 0 → nothing to do
	MOVD $0, R5
ars_j:
	CMP R3, R5
	BGE ars_done
	MOVD R5, R6
	MUL R4, R6
	LSL $3, R6, R6
	MOVD R2, R7
	ADD R6, R7, R7
	MOVD R1, R8
	MOVD R4, R9
	FMOVD $0.0, F0
	CBZ R9, ars_dot_done                      // headDim == 0 → skip dot loop
	VEOR V0.B16, V0.B16, V0.B16
	LSR  $1, R9, R10
	CBZ  R10, ars_dot_tail
ars_dot_pair:
	VLD1.P 16(R8), [V1.D2]
	VLD1.P 16(R7), [V2.D2]
	VFMUL_D2(1, 2, 3)
	VFADD_D2(3, 0, 0)
	SUBS $1, R10, R10
	BNE  ars_dot_pair

	MOVD RSP, R10
	VST1.P [V0.D2], 16(R10)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0

ars_dot_tail:
	TST $1, R9
	BEQ ars_dot_done
	FMOVD (R8), F1
	FMOVD (R7), F2
	FMADDD F1, F0, F2, F0
ars_dot_done:
	FMULD F20, F0, F0
	MOVD R5, R6
	LSL $3, R6, R6
	MOVD R0, R7
	ADD R6, R7, R7
	FMOVD F0, (R7)
	ADD $1, R5, R5
	B ars_j
ars_done:
	RET

// attentionRowOutputNEON computes:
//   out[d] = Σ_j scores[j] * V[j*headDim + d]   for d in 0..headDim
//
// Guards: out is cleared only when headDim > 0; outer loop skipped when
// seqLen == 0.
TEXT ·attentionRowOutputNEON(SB), NOSPLIT, $0-88
	MOVD out+0(FP), R0
	MOVD scores+24(FP), R1
	MOVD V+48(FP), R2
	MOVD seqLen+72(FP), R3
	MOVD headDim+80(FP), R4

	MOVD R4, R9
	MOVD R0, R7
	CBZ R9, aro_clear_done                    // headDim == 0 → nothing to clear
	VEOR V0.B16, V0.B16, V0.B16
	LSR  $1, R9, R10
	CBZ  R10, aro_clear_tail
aro_clear_pair:
	VST1.P [V0.D2], 16(R7)
	SUBS $1, R10, R10
	BNE  aro_clear_pair
aro_clear_tail:
	TST $1, R9
	BEQ aro_clear_done
	FMOVD $0.0, F0
	FMOVD F0, (R7)
aro_clear_done:

	CBZ R3, aro_done                          // seqLen == 0 → no accumulation
	MOVD $0, R5
aro_j:
	CMP R3, R5
	BGE aro_done
	MOVD R5, R6
	MUL R4, R6
	LSL $3, R6, R6
	MOVD R2, R7
	ADD R6, R7, R7
	MOVD R0, R8

	MOVD R5, R6
	LSL $3, R6, R6
	MOVD R1, R9
	ADD R6, R9, R9
	VLD1R (R9), [V16.D2]

	MOVD R4, R9
	CBZ R9, aro_w_done                        // headDim == 0 → skip inner loop
	LSR  $1, R9, R10
	CBZ  R10, aro_w_tail
aro_w_pair:
	VLD1.P 16(R8), [V0.D2]
	VLD1.P 16(R7), [V1.D2]
	VFMUL_D2(16, 1, 2)
	VFADD_D2(2, 0, 0)
	SUB  $16, R8, R8
	VST1.P [V0.D2], 16(R8)
	SUBS $1, R10, R10
	BNE  aro_w_pair
aro_w_tail:
	TST $1, R9
	BEQ aro_w_done
	MOVD R5, R6
	LSL $3, R6, R6
	MOVD R1, R11
	ADD R6, R11, R11
	FMOVD (R11), F16
	FMOVD (R8), F0
	FMOVD (R7), F1
	FMADDD F16, F0, F1, F0
	FMOVD F0, (R8)
aro_w_done:
	ADD $1, R5, R5
	B aro_j
aro_done:
	RET
