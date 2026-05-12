#include "textflag.h"

// attentionRowScoresNEON computes the per-row attention score vector:
//   scores[j] = scale * Σ_d q[d] * K[j*headDim + d]   for j in 0..seqLen
//
// The Go ARM64 toolchain does not accept double-precision `.2D` NEON
// mnemonics, so the inner dot-product uses scalar FP64 instructions
// (FMOVD/FMADDD) — one float64 per cycle. The kernel still lives in
// assembly so the data path never returns to scalar Go.
//
// Guards: returns immediately for seqLen=0; per-row, returns 0 for headDim=0.
//
// ABI0 layout:
//   scores+0(FP)..23, q+24(FP)..47, K+48(FP)..71,
//   seqLen+72(FP), headDim+80(FP), scale+88(FP)
TEXT ·attentionRowScoresNEON(SB), NOSPLIT, $0-96
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
ars_dot:
	FMOVD (R8), F1
	FMOVD (R7), F2
	FMADDD F1, F0, F2, F0
	ADD $8, R8, R8
	ADD $8, R7, R7
	SUBS $1, R9, R9
	BNE ars_dot
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
// Scalar FP64 inner loop (see comment on attentionRowScoresNEON for the
// toolchain constraint). Guards: out is cleared only when headDim > 0;
// outer loop skipped when seqLen == 0.
TEXT ·attentionRowOutputNEON(SB), NOSPLIT, $0-88
	MOVD out+0(FP), R0
	MOVD scores+24(FP), R1
	MOVD V+48(FP), R2
	MOVD seqLen+72(FP), R3
	MOVD headDim+80(FP), R4

	MOVD R4, R9
	MOVD R0, R7
	FMOVD $0.0, F0
	CBZ R9, aro_clear_done                    // headDim == 0 → nothing to clear
aro_clear:
	FMOVD F0, (R7)
	ADD $8, R7, R7
	SUBS $1, R9, R9
	BNE aro_clear
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
	FMOVD (R9), F16                           // caller-saved register

	MOVD R4, R9
	CBZ R9, aro_w_done                        // headDim == 0 → skip inner loop
aro_w:
	FMOVD (R8), F0
	FMOVD (R7), F1
	FMADDD F16, F0, F1, F0
	FMOVD F0, (R8)
	ADD $8, R8, R8
	ADD $8, R7, R7
	SUBS $1, R9, R9
	BNE aro_w
aro_w_done:
	ADD $1, R5, R5
	B aro_j
aro_done:
	RET
