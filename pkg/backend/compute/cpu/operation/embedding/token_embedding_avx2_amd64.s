#include "textflag.h"

// tokenEmbeddingCopyAVX2(dst, src []float64)
// ABI0: dst+0(FP)=ptr, dst_len+8(FP)=len, dst_cap+16(FP)=cap,
//       src_base+24(FP)=ptr, src_len+32(FP)=len, src_cap+40(FP)=cap
//
// Copies src into dst 4 float64 (32 bytes) per iteration using VMOVUPD.
TEXT ·tokenEmbeddingCopyAVX2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX        // dst ptr
	MOVQ src_base+24(FP), SI  // src ptr
	MOVQ src_len+32(FP), BX   // number of float64 elements
	CMPQ BX, $0
	JLE  done

loop4:
	CMPQ BX, $4
	JL   tail
	VMOVUPD (SI), Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, SI
	ADDQ $32, AX
	SUBQ $4, BX
	JMP  loop4

tail:
	CMPQ BX, $2
	JL   tail1
	MOVUPD (SI), X0
	MOVUPD X0, (AX)
	ADDQ $16, SI
	ADDQ $16, AX
	SUBQ $2, BX

tail1:
	CMPQ BX, $1
	JL   done
	MOVQ (SI), CX
	MOVQ CX, (AX)

done:
	VZEROUPPER
	RET
