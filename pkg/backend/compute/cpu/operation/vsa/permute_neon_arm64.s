#include "textflag.h"

// permuteCopyNEON(dst, src []float64)
TEXT ·permuteCopyNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src_base+24(FP), R1
	MOVD src_len+32(FP), R2
	CBZ  R2, done

	LSR $1, R2, R3
	CBZ R3, tail

pairloop:
	VLD1.P 16(R1), [V0.D2]
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R3, R3
	BNE  pairloop

tail:
	AND $1, R2, R4
	CBZ R4, done
	FMOVD (R1), F0
	FMOVD F0, (R0)

done:
	RET
