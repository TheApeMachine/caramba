#include "textflag.h"

// ApplyMaskAVX2(dst, scores, mask []float64)
// ABI0: dst+0(FP), ..., scores+24(FP), ..., mask+48(FP), mask_len+56(FP), ...
TEXT ·ApplyMaskAVX2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP),      AX
	MOVQ scores_len+32(FP), BX
	MOVQ scores+24(FP),  DI
	MOVQ mask+48(FP),    SI
	CMPQ BX, $4
	JL   am_avx2_tail

am_avx2_loop:
	VMOVUPD (DI), Y0
	VMOVUPD (SI), Y1
	VADDPD  Y1, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  am_avx2_loop

am_avx2_tail:
	CMPQ BX, $2
	JL   am_avx2_scalar
	MOVUPD (DI), X0
	MOVUPD (SI), X1
	ADDPD  X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, BX

am_avx2_scalar:
	CMPQ BX, $0
	JLE  am_avx2_done
	MOVSD (DI), X0
	MOVSD (SI), X1
	ADDSD X1, X0
	MOVSD X0, (AX)

am_avx2_done:
	VZEROUPPER
	RET
