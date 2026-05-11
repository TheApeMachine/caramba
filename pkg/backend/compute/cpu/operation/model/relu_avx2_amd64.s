#include "textflag.h"

// reluAVX2(dst []float64) — in-place ReLU, 4-wide AVX2
// ABI0: dst+0(FP)=ptr, dst_len+8(FP)=len, dst_cap+16(FP)=cap
TEXT ·reluAVX2(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP),     AX   // ptr
	MOVQ dst_len+8(FP), BX   // len

	VXORPD Y14, Y14, Y14     // zero

loop4:
	CMPQ BX, $4
	JL   tail1

	VMOVUPD (AX), Y0
	VMAXPD  Y14, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ    $32, AX
	SUBQ    $4, BX
	JMP     loop4

tail1:
	CMPQ BX, $0
	JLE  done

	VMOVSD (AX), X0
	VMAXSD X14, X0, X0   // X14 is already zero (lower 128b of Y14)
	VMOVSD X0, (AX)
	ADDQ   $8, AX
	DECQ   BX
	JMP    tail1

done:
	VZEROUPPER
	RET

// reluSSE2(dst []float64) — in-place ReLU, 2-wide SSE2
TEXT ·reluSSE2(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP),     AX
	MOVQ dst_len+8(FP), BX

	XORPD X14, X14

loop2:
	CMPQ BX, $2
	JL   tail2

	MOVUPD (AX), X0
	MAXPD  X14, X0
	MOVUPD X0, (AX)
	ADDQ   $16, AX
	SUBQ   $2, BX
	JMP    loop2

tail2:
	CMPQ BX, $0
	JLE  done2

	MOVSD (AX), X0
	MAXSD X14, X0   // X14 is already zero
	MOVSD X0, (AX)
	ADDQ  $8, AX
	DECQ  BX
	JMP   tail2

done2:
	RET
