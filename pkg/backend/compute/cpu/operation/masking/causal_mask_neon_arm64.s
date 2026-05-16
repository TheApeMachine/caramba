#include "textflag.h"

// CausalMaskNEON(dst []float64, seqLen int)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP), seqLen+24(FP)
//
// For each row i: write (i+1) zeros then (seqLen-i-1) -Inf values.

DATA ·neonZero+0(SB)/8, $0.0
GLOBL ·neonZero(SB), RODATA|NOPTR, $8

TEXT ·CausalMaskNEON(SB), NOSPLIT, $8-32
	MOVD dst+0(FP), R0        // R0 = dst ptr
	MOVD seqLen+24(FP), R1    // R1 = seqLen
	CBZ  R1, cm_neon_done

	MOVD $0x7FF0000000000000, R6
	MOVD $1, R7
	LSL  $63, R7, R7
	ORR  R7, R6, R6
	FMOVD R6, F31               // F31 = -Inf
	FMOVD ·neonZero(SB), F30    // F30 = 0.0
	VEOR V30.B16, V30.B16, V30.B16
	FMOVD F31, 0(RSP)
	VLD1R (RSP), [V31.D2]

	MOVD ZR, R2                 // R2 = row i = 0

cm_neon_row:
	CMP R1, R2
	BGE cm_neon_done

	// Write (i+1) zeros
	ADD  $1, R2, R3              // R3 = i+1
	MOVD R0, R4                  // R4 = write ptr

	MOVD R3, R8
	LSR  $1, R8, R9
	CBZ  R9, cm_neon_zero_tail
cm_neon_zero_loop:
	VST1.P [V30.D2], 16(R4)
	SUBS $1, R9, R9
	BNE  cm_neon_zero_loop

cm_neon_zero_tail:
	TST  $1, R8
	BEQ  cm_neon_zero_done
	FMOVD F30, (R4)
	ADD  $8, R4

cm_neon_zero_done:
	// Write (seqLen - i - 1) -Inf
	SUB  R2, R1, R3              // R3 = seqLen - i
	SUB  $1, R3                  // R3 = seqLen - i - 1

	MOVD R3, R8
	LSR  $1, R8, R9
	CBZ  R9, cm_neon_inf_tail
cm_neon_inf_loop:
	VST1.P [V31.D2], 16(R4)
	SUBS $1, R9, R9
	BNE  cm_neon_inf_loop

cm_neon_inf_tail:
	TST  $1, R8
	BEQ  cm_neon_row_done
	FMOVD F31, (R4)

cm_neon_row_done:
	// Advance R0 by seqLen * 8
	LSL  $3, R1, R5
	ADD  R5, R0
	ADD  $1, R2
	B    cm_neon_row

cm_neon_done:
	RET
