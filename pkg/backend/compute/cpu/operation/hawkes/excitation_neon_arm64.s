#include "textflag.h"

DATA ·hexLog2E+0(SB)/8, $1.4426950408889634
GLOBL ·hexLog2E(SB), RODATA, $8
DATA ·hexLn2Hi+0(SB)/8, $0.6931471803691864
GLOBL ·hexLn2Hi(SB), RODATA, $8
DATA ·hexLn2Lo+0(SB)/8, $1.9082149292705877e-10
GLOBL ·hexLn2Lo(SB), RODATA, $8
DATA ·hexMaxArg+0(SB)/8, $709.0
GLOBL ·hexMaxArg(SB), RODATA, $8
DATA ·hexMinArg+0(SB)/8, $-708.0
GLOBL ·hexMinArg(SB), RODATA, $8
DATA ·hexC0+0(SB)/8, $1.0
GLOBL ·hexC0(SB), RODATA, $8
DATA ·hexC1+0(SB)/8, $1.0
GLOBL ·hexC1(SB), RODATA, $8
DATA ·hexC2+0(SB)/8, $0.5
GLOBL ·hexC2(SB), RODATA, $8
DATA ·hexC3+0(SB)/8, $0.16666666666666666
GLOBL ·hexC3(SB), RODATA, $8
DATA ·hexC4+0(SB)/8, $0.041666666666666664
GLOBL ·hexC4(SB), RODATA, $8
DATA ·hexC5+0(SB)/8, $0.008333333333333333
GLOBL ·hexC5(SB), RODATA, $8
DATA ·hexC6+0(SB)/8, $0.001388888888888889
GLOBL ·hexC6(SB), RODATA, $8
DATA ·hexC7+0(SB)/8, $0.0001984126984126984
GLOBL ·hexC7(SB), RODATA, $8
DATA ·hexC8+0(SB)/8, $2.4801587301587302e-5
GLOBL ·hexC8(SB), RODATA, $8
DATA ·hexC9+0(SB)/8, $2.7557319223985893e-6
GLOBL ·hexC9(SB), RODATA, $8
DATA ·hexC10+0(SB)/8, $2.7557319223985894e-7
GLOBL ·hexC10(SB), RODATA, $8
DATA ·hexC11+0(SB)/8, $2.5052108385441718e-8
GLOBL ·hexC11(SB), RODATA, $8

// hawkesExcitationNEON returns alpha * Σ exp(-beta*(now - events[i])).
//
// NOTE on the "NEON" suffix:
// The Go ARM64 assembler does not accept double-precision .2D vector
// mnemonics (FMUL.2D, FADD.2D, FMLA.2D, FRINTN.2D, FCVTNS.2D, FMIN.2D,
// FMAX.2D), so the implementation uses scalar FP64 (FMOVD/FMADDD/etc.)
// rather than packed-double SIMD. True NEON vectorisation would require
// hand-encoded WORD directives. The symbol name is kept "NEON" for parity
// with the AVX2/SSE2 dispatch on amd64. All 12 polynomial coefficients are preloaded into FP registers
// before the loop to avoid per-iteration memory broadcasts.
//
// Register map:
//   F0/F1/F2/F3 : loop scratch (f, r, poly-acc, temp)
//   F4          : 2^r bit-cast scratch (loop)
//   F5..F15,F17 : polynomial coefficients C11..C0 (12 regs)
//   F16         : Σ acc (sum)
//   F18         : 1.0 (used after the polynomial)
//   F20..F26    : range-reduction constants
//   F19,F27..F31: unused
TEXT ·hawkesExcitationNEON(SB), NOSPLIT, $0-56
	MOVD events+0(FP), R0
	MOVD events_len+8(FP), R1
	CBZ R1, hex_done

	FMOVD now+24(FP), F20
	FMOVD beta+32(FP), F21
	FMOVD ·hexLog2E(SB), F22
	FMOVD ·hexLn2Hi(SB), F23
	FMOVD ·hexLn2Lo(SB), F24
	FMOVD ·hexMaxArg(SB), F25
	FMOVD ·hexMinArg(SB), F26

	// Hoist polynomial coefficients (C11 high-degree first).
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

	FMOVD $0.0, F16
	MOVD  $1023, R10

hex_loop:
	FMOVD (R0), F0
	FSUBD F0, F20, F0                          // now - event
	FMULD F21, F0, F0                          // *beta
	FNEGD F0, F0                               // → -beta*(now-event)
	FMINNMD F25, F0, F0
	FMAXNMD F26, F0, F0

	FMULD F22, F0, F1
	FRINTND F1, F1
	FMSUBD F1, F0, F23, F0
	FMSUBD F1, F0, F24, F0

	// Horner: y = y*f + Cn using preloaded coefficients
	FMOVD  F5, F2                              // y = C11
	FMADDD F2, F6, F0, F2                      // y = y*f + C10
	FMADDD F2, F7, F0, F2                      // y = y*f + C9
	FMADDD F2, F8, F0, F2                      // y = y*f + C8
	FMADDD F2, F9, F0, F2                      // y = y*f + C7
	FMADDD F2, F10, F0, F2                     // y = y*f + C6
	FMADDD F2, F11, F0, F2                     // y = y*f + C5
	FMADDD F2, F12, F0, F2                     // y = y*f + C4
	FMADDD F2, F13, F0, F2                     // y = y*f + C3
	FMADDD F2, F14, F0, F2                     // y = y*f + C2
	FMADDD F2, F15, F0, F2                     // y = y*f + C1
	FMADDD F2, F17, F0, F2                     // y = y*f + C0

	FCVTZSD F1, R5
	ADD R10, R5, R5
	LSL $52, R5, R5
	FMOVD R5, F4
	FMULD F4, F2, F2
	FADDD F2, F16, F16

	ADD $8, R0, R0
	SUBS $1, R1, R1
	BNE hex_loop

	FMOVD alpha+40(FP), F0
	FMULD F0, F16, F16
	FMOVD F16, ret+48(FP)
	RET

hex_done:
	FMOVD $0.0, F0
	FMOVD F0, ret+48(FP)
	RET
