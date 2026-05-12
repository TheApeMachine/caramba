#include "textflag.h"

// NOTE: scalar FP64 — see hawkes/excitation_neon_arm64.s for the toolchain
// constraint on double-precision NEON .2D mnemonics. Symbol kept "NEON" for
// dispatch-table parity with the amd64 AVX2/SSE2 variants.

// hawkesKernelRowNEON writes out[i] = alpha * exp(-beta * (events[i] - ti)).
// Scalar-FP64 inner loop with all 12 polynomial coefficients preloaded into
// FP registers (NEON has 32 FP regs, well within budget).
//
// Register map:
//   F0/F1/F2/F3 : loop scratch (f, r, poly-acc, temp)
//   F4          : 2^r bit-cast scratch
//   F5..F15,F17 : polynomial coefficients C11..C0
//   F18         : reserved
//   F20..F23    : ti, beta, log2e, ln2hi
//   F24..F27    : ln2lo, maxArg, minArg, alpha
TEXT ·hawkesKernelRowNEON(SB), NOSPLIT, $0-72
	MOVD out+0(FP), R0
	MOVD events+24(FP), R1
	MOVD out_len+8(FP), R2
	CBZ R2, hkr_done

	FMOVD ti+48(FP), F20
	FMOVD beta+64(FP), F21
	FMOVD ·hexLog2E(SB), F22
	FMOVD ·hexLn2Hi(SB), F23
	FMOVD ·hexLn2Lo(SB), F24
	FMOVD ·hexMaxArg(SB), F25
	FMOVD ·hexMinArg(SB), F26
	FMOVD alpha+56(FP), F27

	// Hoist polynomial coefficients (C11..C0).
	FMOVD ·hexC11(SB), F5
	FMOVD ·hexC10(SB), F6
	FMOVD ·hexC9(SB), F7
	FMOVD ·hexC8(SB), F8
	FMOVD ·hexC7(SB), F9
	FMOVD ·hexC6(SB), F10
	FMOVD ·hexC5(SB), F11
	FMOVD ·hexC4(SB), F12
	FMOVD ·hexC3(SB), F13
	FMOVD ·hexC2(SB), F14
	FMOVD ·hexC1(SB), F15
	FMOVD ·hexC0(SB), F17

	MOVD $1023, R10

hkr_loop:
	FMOVD (R1), F0
	FSUBD F20, F0, F0                          // events - ti
	FMULD F21, F0, F0                          // *beta
	FNEGD F0, F0                               // -beta*(events-ti)
	FMINNMD F25, F0, F0
	FMAXNMD F26, F0, F0

	FMULD F22, F0, F1
	FRINTND F1, F1
	FMSUBD F1, F0, F23, F0
	FMSUBD F1, F0, F24, F0

	FMOVD  F5, F2                              // y = C11
	FMADDD F2, F6, F0, F2
	FMADDD F2, F7, F0, F2
	FMADDD F2, F8, F0, F2
	FMADDD F2, F9, F0, F2
	FMADDD F2, F10, F0, F2
	FMADDD F2, F11, F0, F2
	FMADDD F2, F12, F0, F2
	FMADDD F2, F13, F0, F2
	FMADDD F2, F14, F0, F2
	FMADDD F2, F15, F0, F2
	FMADDD F2, F17, F0, F2

	FCVTZSD F1, R5
	ADD R10, R5, R5
	LSL $52, R5, R5
	FMOVD R5, F4
	FMULD F4, F2, F2
	FMULD F27, F2, F2                          // *alpha
	FMOVD F2, (R0)

	ADD $8, R0, R0
	ADD $8, R1, R1
	SUBS $1, R2, R2
	BNE hkr_loop

hkr_done:
	RET
