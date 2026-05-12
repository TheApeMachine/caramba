#include "textflag.h"

// attentionRowScoresNEON(scores, q, K []float64, seqLen, headDim int, scale float64)
TEXT ·attentionRowScoresNEON(SB), NOSPLIT, $0-104
	MOVD scores+0(FP), R0
	MOVD q+24(FP), R1
	MOVD K+48(FP), R2
	MOVD seqLen+72(FP), R3
	MOVD headDim+80(FP), R4
	FMOVD scale+88(FP), F20

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
ars_dot:
	FMOVD (R8), F1
	FMOVD (R7), F2
	FMADDD F1, F0, F2, F0
	ADD $8, R8, R8
	ADD $8, R7, R7
	SUBS $1, R9, R9
	BNE ars_dot
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

// attentionRowOutputNEON(out, scores, V []float64, seqLen, headDim int)
TEXT ·attentionRowOutputNEON(SB), NOSPLIT, $0-88
	MOVD out+0(FP), R0
	MOVD scores+24(FP), R1
	MOVD V+48(FP), R2
	MOVD seqLen+72(FP), R3
	MOVD headDim+80(FP), R4

	MOVD R4, R9
	MOVD R0, R7
	FMOVD $0.0, F0
aro_clear:
	FMOVD F0, (R7)
	ADD $8, R7, R7
	SUBS $1, R9, R9
	BNE aro_clear

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
	FMOVD (R9), F14

	MOVD R4, R9
aro_w:
	FMOVD (R8), F0
	FMOVD (R7), F1
	FMADDD F14, F0, F1, F0
	FMOVD F0, (R8)
	ADD $8, R8, R8
	ADD $8, R7, R7
	SUBS $1, R9, R9
	BNE aro_w
	ADD $1, R5, R5
	B aro_j
aro_done:
	RET
