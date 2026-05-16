#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// layerNormRowNEON(out, row, weight, bias []float64, eps float64)
// Single LayerNorm row, fully fused.
TEXT ·layerNormRowNEON(SB), NOSPLIT, $32-104
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
	VEOR V0.B16, V0.B16, V0.B16
	LSR  $1, R8, R9
	CBZ  R9, ln_sum_tail
ln_sum_loop:
	VLD1.P 16(R7), [V1.D2]
	VFADD_D2(1, 0, 0)
	SUBS $1, R9, R9
	BNE  ln_sum_loop

	MOVD RSP, R9
	VST1.P [V0.D2], 16(R9)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0

ln_sum_tail:
	TST $1, R8
	BEQ ln_post_sum
	FMOVD (R7), F1
	FADDD F1, F0, F0
ln_post_sum:
	SCVTFD R5, F4                              // d as double
	FDIVD F4, F0, F0                           // mean
	FMOVD F0, 16(RSP)
	ADD $16, RSP, R9
	VLD1R (R9), [V20.D2]

	// variance pass
	MOVD R6, R7
	MOVD R4, R8
	FMOVD $0.0, F2
	CBZ R8, ln_post_var
	VEOR V2.B16, V2.B16, V2.B16
	LSR  $1, R8, R9
	CBZ  R9, ln_var_tail
ln_var_loop:
	VLD1.P 16(R7), [V1.D2]
	VFSUB_D2(20, 1, 1)
	VFMUL_D2(1, 1, 3)
	VFADD_D2(3, 2, 2)
	SUBS $1, R9, R9
	BNE  ln_var_loop

	MOVD RSP, R9
	VST1.P [V2.D2], 16(R9)
	FMOVD 0(RSP), F2
	FMOVD 8(RSP), F1
	FADDD F1, F2, F2

ln_var_tail:
	TST $1, R8
	BEQ ln_post_var
	FMOVD (R7), F1
	FSUBD F0, F1, F1                            // x - mean
	FMADDD F1, F2, F1, F2                       // var += diff²
ln_post_var:
	FDIVD F4, F2, F2                            // var / d
	FMOVD eps+96(FP), F5
	FADDD F5, F2, F2
	FSQRTD F2, F2
	FMOVD $1.0, F6
	FDIVD F2, F6, F6                            // invStd
	FMOVD F6, 24(RSP)
	ADD $24, RSP, R9
	VLD1R (R9), [V21.D2]

	// normalize + affine
	MOVD R6, R7
	MOVD R4, R8
	CBZ R8, ln_done
	LSR $1, R8, R9
	CBZ R9, ln_norm_tail
ln_norm_loop:
	VLD1.P 16(R7), [V1.D2]
	VLD1.P 16(R2), [V2.D2]
	VLD1.P 16(R3), [V3.D2]
	VFSUB_D2(20, 1, 1)
	VFMUL_D2(21, 1, 1)
	VFMUL_D2(2, 1, 1)
	VFADD_D2(3, 1, 1)
	VST1.P [V1.D2], 16(R0)
	SUBS $1, R9, R9
	BNE  ln_norm_loop

ln_norm_tail:
	TST $1, R8
	BEQ ln_done
	FMOVD (R7), F1
	FSUBD F0, F1, F1
	FMULD F6, F1, F1
	FMOVD (R2), F3
	FMULD F3, F1, F1
	FMOVD (R3), F3
	FADDD F3, F1, F1
	FMOVD F1, (R0)
ln_done:
	RET
