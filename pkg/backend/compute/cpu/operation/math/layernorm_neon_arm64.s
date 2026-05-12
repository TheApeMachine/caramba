#include "textflag.h"

// layerNormRowNEON(out, row, weight, bias []float64, eps float64)
// Single LayerNorm row, fully fused.
TEXT ·layerNormRowNEON(SB), NOSPLIT, $0-104
	MOVD out+0(FP), R0
	MOVD row+24(FP), R1
	MOVD weight+48(FP), R2
	MOVD bias+72(FP), R3
	MOVD out_len+8(FP), R4

	MOVD R4, R5                                // d
	MOVD R1, R6                                // save row ptr
	MOVD R1, R7                                // working

	FMOVD $0.0, F0                             // sum
	MOVD R4, R8

	CBZ R8, ln_post_sum
ln_sum_loop:
	FMOVD (R7), F1
	FADDD F1, F0, F0
	ADD $8, R7, R7
	SUBS $1, R8, R8
	BNE ln_sum_loop
ln_post_sum:
	SCVTFD R5, F4                              // d as double
	FDIVD F4, F0, F0                           // mean

	// variance pass
	MOVD R6, R7
	MOVD R4, R8
	FMOVD $0.0, F2
	CBZ R8, ln_post_var
ln_var_loop:
	FMOVD (R7), F1
	FSUBD F0, F1, F1                            // x - mean
	FMADDD F1, F2, F1, F2                       // var += diff²
	ADD $8, R7, R7
	SUBS $1, R8, R8
	BNE ln_var_loop
ln_post_var:
	FDIVD F4, F2, F2                            // var / d
	FMOVD eps+96(FP), F5
	FADDD F5, F2, F2
	FSQRTD F2, F2
	FMOVD $1.0, F6
	FDIVD F2, F6, F6                            // invStd

	// normalize + affine
	MOVD R6, R7
	MOVD R4, R8
	CBZ R8, ln_done
ln_norm_loop:
	FMOVD (R7), F1
	FSUBD F0, F1, F1
	FMULD F6, F1, F1
	FMOVD (R2), F3
	FMULD F3, F1, F1
	FMOVD (R3), F3
	FADDD F3, F1, F1
	FMOVD F1, (R0)
	ADD $8, R0, R0
	ADD $8, R7, R7
	ADD $8, R2, R2
	ADD $8, R3, R3
	SUBS $1, R8, R8
	BNE ln_norm_loop
ln_done:
	RET
