#include "textflag.h"

#define VFMAXNM_D2(m, n, d) WORD $(0x4E60C400 | ((m) << 16) | ((n) << 5) | (d))

// argmaxNEON(xs []float64) int
// Returns the index of the largest element, or 0 for an empty slice.
// NaN values never displace an existing best (matches scalar fallback).
TEXT ·argmaxNEON(SB), NOSPLIT, $16-32
	MOVD xs+0(FP), R0                         // base pointer
	MOVD xs_len+8(FP), R1
	MOVD $0, R2                               // result index, default 0
	CBZ R1, am_done

	FMOVD (R0), F0
	FCMPD F0, F0
	BNE am_done                              // scalar contract: leading NaN keeps index 0

	CMP $1, R1
	BEQ am_found                              // single element

	VLD1.P 16(R0), [V0.D2]
	MOVD R1, R3
	LSR  $1, R3, R4
	SUBS $1, R4, R4
	BEQ  am_reduce

am_vloop:
	VLD1.P 16(R0), [V1.D2]
	VFMAXNM_D2(1, 0, 0)
	SUBS $1, R4, R4
	BNE  am_vloop

am_reduce:
	MOVD RSP, R5
	VST1.P [V0.D2], 16(R5)
	FMOVD 0(RSP), F0
	FMOVD 8(RSP), F1
	FCMPD F0, F1
	BLE am_tail
	FMOVD F1, F0

am_tail:
	TST $1, R1
	BEQ am_scan
	MOVD R1, R5
	SUB  $1, R5, R5
	LSL  $3, R5, R5
	MOVD xs+0(FP), R6
	ADD  R5, R6, R6
	FMOVD (R6), F2
	FCMPD F0, F2
	BLE am_scan
	FMOVD F2, F0

am_scan:
	MOVD xs+0(FP), R6
	MOVD $0, R5
am_scan_loop:
	FMOVD (R6), F2
	FCMPD F0, F2
	BEQ  am_scan_found
	ADD  $8, R6, R6
	ADD  $1, R5, R5
	CMP  R1, R5
	BLT  am_scan_loop
	B    am_done

am_scan_found:
	MOVD R5, R2

am_found:
am_done:
	MOVD R2, ret+24(FP)
	RET
