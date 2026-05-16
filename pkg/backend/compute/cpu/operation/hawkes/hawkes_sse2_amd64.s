#include "textflag.h"

// expSumSSE2(expBuf []float64) float64
TEXT ·expSumSSE2(SB), NOSPLIT, $0-32
	MOVQ  expBuf+0(FP), AX
	MOVQ  expBuf_len+8(FP), BX
	XORPS X0, X0
	CMPQ  BX, $2
	JL    done_es_sse

loop_es_sse:
	MOVUPD (AX), X1
	ADDPD  X1, X0
	ADDQ   $16, AX
	SUBQ   $2, BX
	CMPQ   BX, $2
	JGE    loop_es_sse

done_es_sse:
	MOVAPD X0, X1
	UNPCKHPD X0, X1
	ADDSD X1, X0
	TESTQ  BX, BX
	JZ     end_es_sse

scalar_es_sse:
	MOVSD (AX), X1
	ADDSD X1, X0
	ADDQ  $8, AX
	DECQ  BX
	JNZ   scalar_es_sse

end_es_sse:
	MOVSD X0, ret+24(FP)
	RET

// subVecSSE2(dst, a, b []float64)
TEXT ·subVecSSE2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ dst_len+8(FP), BX
	MOVQ a_len+32(FP), CX
	CMPQ CX, BX
	JAE  sub_sv2_min1
	MOVQ CX, BX

sub_sv2_min1:
	MOVQ b_len+56(FP), CX
	CMPQ CX, BX
	JAE  sub_sv2_min2
	MOVQ CX, BX

sub_sv2_min2:
	MOVQ a+24(FP), DI
	MOVQ b+48(FP), SI
	CMPQ BX, $2
	JL   tail_sv2

loop_sv2:
	MOVUPD (DI), X0
	MOVUPD (SI), X1
	SUBPD  X1, X0
	MOVUPD X0, (AX)
	ADDQ   $16, AX
	ADDQ   $16, DI
	ADDQ   $16, SI
	SUBQ   $2, BX
	CMPQ   BX, $2
	JGE    loop_sv2

tail_sv2:
	CMPQ BX, $1
	JNE  done_sv2
	MOVSD (DI), X0
	MOVSD (SI), X1
	SUBSD X1, X0
	MOVSD X0, (AX)

done_sv2:
	RET
