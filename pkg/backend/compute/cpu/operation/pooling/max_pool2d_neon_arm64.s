#include "textflag.h"

// reduceMaxNEON(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceMaxNEON(SB), NOSPLIT, $0-32
	MOVD   a+0(FP), R0
	MOVD   a_len+8(FP), R1
	CBZ    R1, done_rm
	FMOVD.P 8(R0), F0
	SUBS $1, R1, R1
	CBZ  R1, done_rm
loop_rm:
	FMOVD.P 8(R0), F1
	FCMPD   F0, F1
	FCSELD  GT, F0, F1, F0
	SUBS $1, R1, R1
	BNE  loop_rm
done_rm:
	FMOVD F0, ret+24(FP)
	RET
