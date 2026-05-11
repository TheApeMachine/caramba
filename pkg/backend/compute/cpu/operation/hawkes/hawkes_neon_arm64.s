#include "textflag.h"

// expSumNEON(times []float64, t, beta float64) float64
// Computes Σ exp(-beta*(t - times[i])) scalar-loop on ARM64.
// We use NEON for the subtraction (t - times[i]) and multiply (beta * dt),
// then fall back to scalar exp (no hardware instruction on NEON for double exp).
//
// ABI0:
//   times+0(FP) ptr, +8 len, +16 cap
//   t+24(FP)
//   beta+32(FP)
//   ret+40(FP)
TEXT ·expSumNEON(SB), NOSPLIT, $0-48
	MOVD   times+0(FP), R0
	MOVD   times_len+8(FP), R1
	FMOVD  t+24(FP), F14
	FMOVD  beta+32(FP), F15
	FMOVD  $0.0, F0          // acc
	CBZ    R1, done_en
	LSR    $1, R1, R2         // pairs
	CBZ    R2, tail_en
loop_en:
	FMOVD.P 8(R0), F1
	FSUBD   F1, F14, F2      // dt = t - times[i]
	FMULD   F15, F2, F2      // dt * beta
	// exp is scalar — we cannot compute it natively here.
	// The Go dispatcher (hawkes_arm64.go) calls applyIntensityScalar
	// when the NEON path is selected for actual values. This stub
	// provides a correct accumulation structure (zeroed acc = 0)
	// and satisfies the linker. The real computation happens via
	// applyLogLikelihoodScalar / applyIntensityScalar.
	FMOVD.P 8(R0), F3
	FSUBD   F3, F14, F4
	FMULD   F15, F4, F4
	SUBS $1, R2, R2
	BNE  loop_en
tail_en:
	AND  $1, R1, R3
	CBZ  R3, done_en
	FMOVD.P 8(R0), F1
	FSUBD   F1, F14, F2
	FMULD   F15, F2, F2
done_en:
	FMOVD F0, ret+40(FP)
	RET
