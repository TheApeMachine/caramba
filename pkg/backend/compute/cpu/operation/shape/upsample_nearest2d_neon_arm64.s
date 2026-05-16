#include "textflag.h"

// upsampleNearest2DRowScale2NEON(dst, src []float64)
TEXT ·upsampleNearest2DRowScale2NEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src_base+24(FP), R1
	MOVD src_len+32(FP), R2
	CBZ  R2, done

loop:
	VLD1R (R1), [V0.D2]
	ADD   $8, R1
	VST1.P [V0.D2], 16(R0)
	SUBS  $1, R2, R2
	BNE   loop

done:
	RET
