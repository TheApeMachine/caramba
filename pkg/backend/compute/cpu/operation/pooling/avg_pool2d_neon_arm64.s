#include "textflag.h"

// reduceSumNEON(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceSumNEON(SB), NOSPLIT, $0-32
	MOVD   a+0(FP), R0
	MOVD   a_len+8(FP), R1
	FMOVD  $0.0, F0
	LSR    $1, R1, R2
	CBZ    R2, done_rs
loop_rs:
	FMOVD.P 8(R0), F1
	FADDD   F1, F0, F0
	FMOVD.P 8(R0), F2
	FADDD   F2, F0, F0
	SUBS $1, R2, R2
	BNE  loop_rs
done_rs:
	FMOVD F0, ret+24(FP)
	RET
