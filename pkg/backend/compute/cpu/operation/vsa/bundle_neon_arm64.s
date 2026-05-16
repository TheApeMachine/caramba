#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// bundleAccumNEON(dst, src []float64)
TEXT ·bundleAccumNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD dst_len+8(FP), R2
	CBZ R2, ba_done
	LSR $1, R2, R3
	CBZ R3, ba_tail
ba_loop:
	VLD1.P 16(R0), [V0.D2]
	VLD1.P 16(R1), [V1.D2]
	VFADD_D2(1, 0, 0)
	SUB $16, R0, R0
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R3, R3
	BNE ba_loop
ba_tail:
	TST $1, R2
	BEQ ba_done
	FMOVD (R0), F0
	FMOVD (R1), F1
	FADDD F1, F0, F0
	FMOVD F0, (R0)
ba_done:
	RET

// bundleNormalizeNEON(dst []float64, eps float64)
TEXT ·bundleNormalizeNEON(SB), NOSPLIT, $24-32
	MOVD dst+0(FP), R0
	MOVD dst_len+8(FP), R1
	FMOVD eps+24(FP), F10
	MOVD R0, R3
	MOVD R1, R4

	FMOVD $0.0, F0
	CBZ R1, bn_done
	VEOR V0.B16, V0.B16, V0.B16
	LSR $1, R1, R5
	CBZ R5, bn_ss_tail
bn_ss_loop:
	VLD1.P 16(R0), [V1.D2]
	VFMUL_D2(1, 1, 2)
	VFADD_D2(2, 0, 0)
	SUBS $1, R5, R5
	BNE bn_ss_loop

	MOVD RSP, R5
	VST1.P [V0.D2], 16(R5)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FADDD F1, F0, F0

bn_ss_tail:
	TST $1, R1
	BEQ bn_post_sum
	FMOVD (R0), F1
	FMADDD F1, F0, F1, F0

bn_post_sum:
	FSQRTD F0, F0
	FCMPD F10, F0
	BLE bn_done
	FMOVD $1.0, F2
	FDIVD F0, F2, F2                            // 1/norm
	FMOVD F2, 16(RSP)
	ADD $16, RSP, R5
	VLD1R (R5), [V20.D2]

	MOVD R3, R0
	MOVD R4, R1
	LSR $1, R1, R5
	CBZ R5, bn_scale_tail
bn_scale_loop:
	VLD1.P 16(R0), [V1.D2]
	VFMUL_D2(20, 1, 1)
	SUB $16, R0, R0
	VST1.P [V1.D2], 16(R0)
	SUBS $1, R5, R5
	BNE bn_scale_loop

bn_scale_tail:
	TST $1, R1
	BEQ bn_done
	FMOVD (R0), F3
	FMULD F2, F3, F3
	FMOVD F3, (R0)

bn_done:
	RET
