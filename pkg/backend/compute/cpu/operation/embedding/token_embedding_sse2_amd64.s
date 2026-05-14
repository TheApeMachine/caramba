#include "textflag.h"

// tokenEmbeddingCopySSE2(dst, src []float64)
// ABI0: dst+0(FP)=ptr, dst_len+8(FP)=len, dst_cap+16(FP)=cap,
//       src_base+24(FP)=ptr, src_len+32(FP)=len, src_cap+40(FP)=cap
//
// Copies src into dst 2 float64 (16 bytes) per iteration using MOVUPD.
TEXT ·tokenEmbeddingCopySSE2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX        // dst ptr
	MOVQ src_base+24(FP), SI  // src ptr
	MOVQ src_len+32(FP), BX   // number of float64 elements
	CMPQ BX, $0
	JLE  done

loop2:
	CMPQ BX, $2
	JL   tail1
	MOVUPD (SI), X0
	MOVUPD X0, (AX)
	ADDQ $16, SI
	ADDQ $16, AX
	SUBQ $2, BX
	JMP  loop2

tail1:
	CMPQ BX, $1
	JL   done
	MOVQ (SI), CX
	MOVQ CX, (AX)

done:
	RET
