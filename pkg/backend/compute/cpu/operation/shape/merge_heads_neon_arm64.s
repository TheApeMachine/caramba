#include "textflag.h"

// mergeHeadsCopyNEON(dst, src []float64)
TEXT ·mergeHeadsCopyNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src_base+24(FP), R1
	MOVD src_len+32(FP), R2
	CBZ  R2, done

	LSR $1, R2, R3
	CBZ R3, tail

pairloop:
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R1), F1
	FMOVD.P F0, 8(R0)
	FMOVD.P F1, 8(R0)
	SUBS $1, R3, R3
	BNE  pairloop

tail:
	AND $1, R2, R4
	CBZ R4, done
	FMOVD (R1), F0
	FMOVD F0, (R0)

done:
	RET
