#include "textflag.h"

// CausalMaskSSE2(dst []float64, seqLen int)
// ABI0: dst+0(FP)=ptr, dst_len+8(FP)=len, dst_cap+16(FP)=cap, seqLen+24(FP)=int

DATA ·sse2NegInf+0(SB)/8, $0xFFF0000000000000   // -Inf
DATA ·sse2NegInf+8(SB)/8, $0xFFF0000000000000   // -Inf
GLOBL ·sse2NegInf(SB), RODATA|NOPTR, $16

TEXT ·CausalMaskSSE2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), AX
	MOVQ seqLen+24(FP), CX
	CMPQ CX, $0
	JLE  cm_sse2_done

	MOVUPD ·sse2NegInf(SB), X15  // X15 = [-Inf, -Inf]
	XORPS  X14, X14              // X14 = [0.0, 0.0]

	XORQ DX, DX   // row i

cm_sse2_row:
	CMPQ DX, CX
	JGE  cm_sse2_done

	MOVQ DX, BX
	INCQ BX        // zeros to write
	MOVQ AX, DI

cm_sse2_zero_loop:
	CMPQ BX, $0
	JLE  cm_sse2_zero_done
	MOVSD X14, (DI)
	ADDQ  $8, DI
	DECQ  BX
	JMP   cm_sse2_zero_loop

cm_sse2_zero_done:
	MOVQ CX, BX
	SUBQ DX, BX
	DECQ BX        // -inf count

cm_sse2_inf_loop:
	CMPQ BX, $2
	JL   cm_sse2_inf_tail
	MOVUPD X15, (DI)
	ADDQ $16, DI
	SUBQ $2, BX
	JMP  cm_sse2_inf_loop

cm_sse2_inf_tail:
	CMPQ BX, $0
	JLE  cm_sse2_row_done
	MOVSD X15, (DI)
	ADDQ  $8, DI
	DECQ  BX
	JMP   cm_sse2_inf_tail

cm_sse2_row_done:
	MOVQ CX, SI
	SHLQ $3, SI
	ADDQ SI, AX
	INCQ DX
	JMP  cm_sse2_row

cm_sse2_done:
	RET
