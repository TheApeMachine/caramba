#include "textflag.h"

// expSumNEON(expBuf []float64) float64
TEXT ·expSumNEON(SB), NOSPLIT, $0-32
	MOVD   expBuf+0(FP), R0
	MOVD   expBuf_len+8(FP), R1
	FMOVD  $0.0, F0
	LSR    $1, R1, R2
	CBZ    R2, done_en
loop_en:
	FMOVD.P 8(R0), F1
	FADDD   F1, F0, F0
	FMOVD.P 8(R0), F2
	FADDD   F2, F0, F0
	SUBS $1, R2, R2
	BNE  loop_en
done_en:
	TST $1, R1
	BEQ store_en
	FMOVD.P 8(R0), F1
	FADDD F1, F0, F0
store_en:
	FMOVD F0, ret+24(FP)
	RET
