#include "textflag.h"

// CopyNEON(dst, src []float64)
// Copies src into dst 2 float64s per iteration using FMOVD.
// ABI0: dst+0(FP)=ptr, dst_len+8(FP)=len, dst_cap+16(FP)=cap,
//       src_base+24(FP)=ptr, src_len+32(FP)=len, src_cap+40(FP)=cap
TEXT ·CopyNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src_base+24(FP), R1
	MOVD src_len+32(FP), R2
	CBZ  R2, done

	// Process 2 float64s per iteration.
	LSR  $1, R2, R3
	CBZ  R3, tail

pairloop:
	FMOVD.P 8(R1), F0
	FMOVD.P 8(R1), F1
	FMOVD.P F0, 8(R0)
	FMOVD.P F1, 8(R0)
	SUBS $1, R3, R3
	BNE  pairloop

tail:
	// Handle odd element if any.
	AND  $1, R2, R4
	CBZ  R4, done
	FMOVD (R1), F0
	FMOVD F0, (R0)

done:
	RET
