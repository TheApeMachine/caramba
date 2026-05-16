#include "textflag.h"

#define VFMAXNM_D2(m, n, d) WORD $(0x4E60C400 | ((m) << 16) | ((n) << 5) | (d))

// reluNEON(dst []float64) — in-place ReLU, 2-wide NEON pairs
// ABI0: dst+0(FP)=ptr, dst_len+8(FP)=len, dst_cap+16(FP)=cap
TEXT ·reluNEON(SB), NOSPLIT, $0-24
	MOVD  dst+0(FP),     R0   // ptr
	MOVD  dst_len+8(FP), R1   // len

	FMOVD ZR, F31             // zero constant via zero register (no DATA/GLOBL needed)
	VEOR  V31.B16, V31.B16, V31.B16

	LSR  $1, R1, R2   // pairs = len / 2
	CBZ  R2, tail

pairloop:
	VLD1.P 16(R0), [V0.D2]
	SUB    $16, R0, R0
	VFMAXNM_D2(31, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS   $1, R2, R2
	BNE    pairloop

tail:
	AND  $1, R1, R3
	CBZ  R3, done

	FMOVD (R0), F0
	FMAXD F31, F0, F0
	FMOVD F0, (R0)

done:
	RET
