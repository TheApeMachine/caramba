#include "textflag.h"

// ApplyMaskNEON(dst, scores, mask []float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP),
//       scores+24(FP), scores_len+32(FP), scores_cap+40(FP),
//       mask+48(FP), mask_len+56(FP), mask_cap+64(FP)
TEXT ·ApplyMaskNEON(SB), NOSPLIT, $0-72
	MOVD dst+0(FP),         R0
	MOVD scores_len+32(FP), R3
	MOVD scores+24(FP),     R1
	MOVD mask+48(FP),       R2
	LSR  $1, R3, R4
	CBZ  R4, am_neon_tail

am_neon_loop:
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R2), F1
	FADDD   F1, F0, F0
	FMOVD.P F0, 8(R0)
	FMOVD.P 8(R1), F2
	FMOVD.P 8(R2), F3
	FADDD   F3, F2, F2
	FMOVD.P F2, 8(R0)
	SUBS $1, R4, R4
	BNE  am_neon_loop

am_neon_tail:
	AND  $1, R3, R5
	CBZ  R5, am_neon_done
	FMOVD (R1), F0
	FMOVD (R2), F1
	FADDD F1, F0, F0
	FMOVD F0, (R0)

am_neon_done:
	RET
