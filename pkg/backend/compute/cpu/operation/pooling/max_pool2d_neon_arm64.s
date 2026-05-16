#include "textflag.h"

#define VFMAXNM_D2(m, n, d) WORD $(0x4E60C400 | ((m) << 16) | ((n) << 5) | (d))

// reduceMaxNEON(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceMaxNEON(SB), NOSPLIT, $16-32
	MOVD   a+0(FP), R0
	MOVD   a_len+8(FP), R1
	CBZ    R1, done_rm
	CMP    $2, R1
	BLT    scalar_rm
	VLD1.P 16(R0), [V0.D2]
	MOVD   R1, R2
	LSR    $1, R2, R3
	SUBS   $1, R3, R3
	BEQ    reduce_rm
loop_rm:
	VLD1.P 16(R0), [V1.D2]
	VFMAXNM_D2(1, 0, 0)
	SUBS $1, R3, R3
	BNE  loop_rm
reduce_rm:
	MOVD RSP, R3
	VST1.P [V0.D2], 16(R3)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FCMPD F0, F1
	BLE tail_rm
	FMOVD F1, F0
tail_rm:
	TST $1, R1
	BEQ done_rm
	MOVD R1, R2
	SUB  $1, R2, R2
	LSL  $3, R2, R2
	MOVD a+0(FP), R3
	ADD  R2, R3, R3
	FMOVD (R3), F1
	FCMPD F0, F1
	BLE done_rm
	FMOVD F1, F0
	B done_rm
scalar_rm:
	FMOVD.P 8(R0), F0
	SUBS $1, R1, R1
	CBZ  R1, done_rm
scalar_loop_rm:
	FMOVD.P 8(R0), F1
	FCMPD   F0, F1
	FCSELD  GT, F0, F1, F0
	SUBS $1, R1, R1
	BNE  scalar_loop_rm
done_rm:
	FMOVD F0, ret+24(FP)
	RET
