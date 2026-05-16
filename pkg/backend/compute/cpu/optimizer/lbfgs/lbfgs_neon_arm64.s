#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// lbfgsSubNEON(dst, a, b []float64)
TEXT ·lbfgsSubNEON(SB), NOSPLIT, $0-72
	MOVD dst+0(FP), R0
	MOVD a+24(FP), R1
	MOVD b+48(FP), R2
	MOVD dst_len+8(FP), R3
	LSR  $1, R3, R4
	CBZ  R4, sub_neon_tail
sub_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R2), [V1.D2]
	VFSUB_D2(1, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R4, R4
	BNE sub_neon_loop
sub_neon_tail:
	TST $1, R3
	BEQ sub_neon_done
	FMOVD (R1), F0
	FMOVD (R2), F1
	FSUBD F1, F0, F0
	FMOVD F0, (R0)
sub_neon_done:
	RET

// lbfgsDotNEON(a, b []float64) float64
TEXT ·lbfgsDotNEON(SB), NOSPLIT, $16-56
	MOVD a+0(FP), R0
	MOVD b+24(FP), R1
	MOVD a_len+8(FP), R2
	VEOR V0.B16, V0.B16, V0.B16
	LSR  $1, R2, R3
	CBZ  R3, dot_neon_tail
dot_neon_loop:
	VLD1.P 16(R0), [V1.D2]
	VLD1.P 16(R1), [V2.D2]
	VFMUL_D2(2, 1, 3)
	VFADD_D2(3, 0, 0)
	SUBS $1, R3, R3
	BNE dot_neon_loop
	MOVD RSP, R3
	VST1.P [V0.D2], 16(R3)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0
dot_neon_tail:
	TST $1, R2
	BEQ dot_neon_done
	FMOVD (R0), F1
	FMOVD (R1), F2
	FMADDD F1, F0, F2, F0
dot_neon_done:
	FMOVD F0, ret+48(FP)
	RET

// lbfgsAddScaledNEON(dst, src []float64, scale float64)
TEXT ·lbfgsAddScaledNEON(SB), NOSPLIT, $8-56
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD dst_len+8(FP), R2
	FMOVD scale+48(FP), F20
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	LSR $1, R2, R3
	CBZ R3, adsc_neon_tail
adsc_neon_loop:
	VLD1 (R0), [V0.D2]
	VLD1.P 16(R1), [V1.D2]
	VFMUL_D2(20, 1, 1)
	VFADD_D2(1, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R3, R3
	BNE adsc_neon_loop
adsc_neon_tail:
	TST $1, R2
	BEQ adsc_neon_done
	FMOVD (R0), F0
	FMOVD (R1), F1
	FMADDD F20, F0, F1, F0
	FMOVD F0, (R0)
adsc_neon_done:
	RET

// lbfgsScaleNEON(dst []float64, scale float64)
TEXT ·lbfgsScaleNEON(SB), NOSPLIT, $8-32
	MOVD dst+0(FP), R0
	MOVD dst_len+8(FP), R1
	FMOVD scale+24(FP), F20
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	LSR $1, R1, R2
	CBZ R2, sc_neon_tail
sc_neon_loop:
	VLD1 (R0), [V0.D2]
	VFMUL_D2(20, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R2, R2
	BNE sc_neon_loop
sc_neon_tail:
	TST $1, R1
	BEQ sc_neon_done
	FMOVD (R0), F0
	FMULD F20, F0, F0
	FMOVD F0, (R0)
sc_neon_done:
	RET

// lbfgsParamStepNEON(out, params, dir []float64, lr float64)
TEXT ·lbfgsParamStepNEON(SB), NOSPLIT, $8-80
	MOVD out+0(FP), R0
	MOVD params+24(FP), R1
	MOVD dir+48(FP), R2
	MOVD out_len+8(FP), R3
	FMOVD lr+72(FP), F20
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	LSR $1, R3, R4
	CBZ R4, ps_neon_tail
ps_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R2), [V1.D2]
	VFMUL_D2(20, 1, 1)
	VFSUB_D2(1, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R4, R4
	BNE ps_neon_loop
ps_neon_tail:
	TST $1, R3
	BEQ ps_neon_done
	FMOVD (R1), F0
	FMOVD (R2), F1
	FMSUBD F20, F0, F1, F0
	FMOVD F0, (R0)
ps_neon_done:
	RET
