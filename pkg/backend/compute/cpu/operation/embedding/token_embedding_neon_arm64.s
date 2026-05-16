#include "textflag.h"

// tokenEmbeddingCopyNEON(dst, src []float64)
// ABI0: dst+0(FP)=ptr, dst_len+8(FP)=len, dst_cap+16(FP)=cap,
//       src_base+24(FP)=ptr, src_len+32(FP)=len, src_cap+40(FP)=cap
//
// Copies src into dst 2 float64 (16 bytes) per iteration.
TEXT ·tokenEmbeddingCopyNEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0        // dst ptr
	MOVD src_base+24(FP), R1  // src ptr
	MOVD src_len+32(FP), R2   // number of float64 elements

	LSR $1, R2, R3            // pairs = n / 2
	CBZ R3, tail

pairloop:
	VLD1.P 16(R1), [V0.D2]
	VST1.P [V0.D2], 16(R0)
	SUBS  $1, R3, R3
	BNE   pairloop

tail:
	// Handle odd element
	AND $1, R2, R4
	CBZ R4, done
	FMOVD (R1), F0
	FMOVD F0, (R0)

done:
	RET
