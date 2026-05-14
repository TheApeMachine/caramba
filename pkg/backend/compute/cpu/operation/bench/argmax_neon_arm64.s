#include "textflag.h"

// argmaxNEON(xs []float64) int
// Returns the index of the largest element, or 0 for an empty slice.
// NaN values never displace an existing best (matches scalar fallback).
//
// Strategy: two independent argmax accumulators run over even/odd indices in
// lock-step so the CPU can issue both compares concurrently. After the paired
// loop the streams are merged and any final element is folded in. This is
// scalar FP64 because the Go ARM64 assembler does not accept double-precision
// `.2D` NEON mnemonics; symbol is kept "NEON" for dispatch-table parity with
// the amd64 AVX2/SSE2 variants (same convention as hawkes/*_neon_arm64.s).
TEXT ·argmaxNEON(SB), NOSPLIT, $0-32
	MOVD xs+0(FP), R0                         // base pointer
	MOVD xs_len+8(FP), R1
	MOVD $0, R2                               // result index, default 0
	CBZ R1, am_done

	FMOVD (R0), F0                            // best_e = xs[0]
	MOVD $0, R3                               // idx_e = 0

	CMP $1, R1
	BEQ am_finish                             // single element

	FMOVD 8(R0), F1                           // best_o = xs[1]
	MOVD $1, R4                               // idx_o = 1
	MOVD $2, R5                               // i
	ADD $16, R0, R6                           // &xs[2]

am_loop:
	ADD $1, R5, R7                            // i+1
	CMP R1, R7
	BGE am_merge                              // need a pair; fewer than 2 left

	FMOVD (R6), F2                            // xs[i]
	FMOVD 8(R6), F3                           // xs[i+1]

	FCMPD F0, F2                              // flags = F2 - F0
	BLE am_skip_e                             // skip on F2 ≤ F0 or NaN
	FMOVD F2, F0
	MOVD R5, R3
am_skip_e:
	FCMPD F1, F3
	BLE am_skip_o
	FMOVD F3, F1
	MOVD R7, R4
am_skip_o:
	ADD $2, R5, R5
	ADD $16, R6, R6
	B am_loop

am_merge:
	FCMPD F0, F1                              // flags = F1 - F0
	BLE am_after_merge                        // F1 ≤ F0 or NaN → keep even
	FMOVD F1, F0
	MOVD R4, R3
am_after_merge:
	CMP R1, R5
	BGE am_finish

	FMOVD (R6), F2
	FCMPD F0, F2
	BLE am_finish
	FMOVD F2, F0
	MOVD R5, R3

am_finish:
	MOVD R3, R2
am_done:
	MOVD R2, ret+24(FP)
	RET
