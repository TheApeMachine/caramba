#include "textflag.h"

// lbfgsSubNEON(dst, a, b []float64)
TEXT ·lbfgsSubNEON(SB), NOSPLIT, $0-72
	MOVD dst+0(FP), R0
	MOVD a+24(FP), R1
	MOVD b+48(FP), R2
	MOVD dst_len+8(FP), R3
	CBZ R3, sub_neon_done
sub_neon_loop:
	FMOVD (R1), F0
	FMOVD (R2), F1
	FSUBD F1, F0, F0
	FMOVD F0, (R0)
	ADD $8, R0, R0
	ADD $8, R1, R1
	ADD $8, R2, R2
	SUBS $1, R3, R3
	BNE sub_neon_loop
sub_neon_done:
	RET

// lbfgsDotNEON(a, b []float64) float64
TEXT ·lbfgsDotNEON(SB), NOSPLIT, $0-56
	MOVD a+0(FP), R0
	MOVD b+24(FP), R1
	MOVD a_len+8(FP), R2
	FMOVD $0.0, F0
	CBZ R2, dot_neon_done
dot_neon_loop:
	FMOVD (R0), F1
	FMOVD (R1), F2
	FMADDD F1, F0, F2, F0
	ADD $8, R0, R0
	ADD $8, R1, R1
	SUBS $1, R2, R2
	BNE dot_neon_loop
dot_neon_done:
	FMOVD F0, ret+48(FP)
	RET

// lbfgsAddScaledNEON(dst, src []float64, scale float64)
TEXT ·lbfgsAddScaledNEON(SB), NOSPLIT, $0-56
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD dst_len+8(FP), R2
	FMOVD scale+48(FP), F20
	CBZ R2, adsc_neon_done
adsc_neon_loop:
	FMOVD (R0), F0
	FMOVD (R1), F1
	FMADDD F20, F0, F1, F0
	FMOVD F0, (R0)
	ADD $8, R0, R0
	ADD $8, R1, R1
	SUBS $1, R2, R2
	BNE adsc_neon_loop
adsc_neon_done:
	RET

// lbfgsScaleNEON(dst []float64, scale float64)
TEXT ·lbfgsScaleNEON(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R0
	MOVD dst_len+8(FP), R1
	FMOVD scale+24(FP), F20
	CBZ R1, sc_neon_done
sc_neon_loop:
	FMOVD (R0), F0
	FMULD F20, F0, F0
	FMOVD F0, (R0)
	ADD $8, R0, R0
	SUBS $1, R1, R1
	BNE sc_neon_loop
sc_neon_done:
	RET

// lbfgsParamStepNEON(out, params, dir []float64, lr float64)
TEXT ·lbfgsParamStepNEON(SB), NOSPLIT, $0-80
	MOVD out+0(FP), R0
	MOVD params+24(FP), R1
	MOVD dir+48(FP), R2
	MOVD out_len+8(FP), R3
	FMOVD lr+72(FP), F20
	CBZ R3, ps_neon_done
ps_neon_loop:
	FMOVD (R1), F0
	FMOVD (R2), F1
	FMSUBD F20, F0, F1, F0
	FMOVD F0, (R0)
	ADD $8, R0, R0
	ADD $8, R1, R1
	ADD $8, R2, R2
	SUBS $1, R3, R3
	BNE ps_neon_loop
ps_neon_done:
	RET
