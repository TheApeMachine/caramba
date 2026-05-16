#include "textflag.h"

// bindSSE2(dst, a, b []float64)
// ABI0: dst+0(FP)..16, a+24(FP)..40, b+48(FP)..64
TEXT ·bindSSE2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ a_len+32(FP), BX
	MOVQ a+24(FP), DI
	MOVQ b+48(FP), SI
	CMPQ BX, $2
	JL   done_bs
loop_bs:
	MOVUPD (DI), X0
	MOVUPD (SI), X1
	MULPD  X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_bs
done_bs:
	RET

// dotReduceSSE2(a, b []float64) float64
// ABI0: a+0(FP)..16, b+24(FP)..40, ret+48(FP)
TEXT ·dotReduceSSE2(SB), NOSPLIT, $0-56
	MOVQ  a+0(FP), AX
	MOVQ  a_len+8(FP), BX
	MOVQ  b+24(FP), DI
	XORPD X0, X0
	CMPQ  BX, $2
	JL    done_ds2
loop_ds2:
	MOVUPD (AX), X1
	MOVUPD (DI), X2
	MULPD  X2, X1
	ADDPD  X1, X0
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_ds2
done_ds2:
	MOVAPD X0, X1
	UNPCKHPD X0, X1
	ADDSD X1, X0
	MOVSD  X0, ret+48(FP)
	RET

// addInPlaceSSE2(dst, src []float64)
// ABI0: dst+0(FP)..16, src+24(FP)..40
TEXT ·addInPlaceSSE2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src_len+32(FP), BX
	MOVQ src+24(FP), DI
	CMPQ BX, $2
	JL   done_aip2
loop_aip2:
	MOVUPD (AX), X0
	MOVUPD (DI), X1
	ADDPD  X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_aip2
done_aip2:
	RET

// mulScalarVecSSE2(dst []float64, s float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP), s+24(FP)
TEXT ·mulScalarVecSSE2(SB), NOSPLIT, $0-32
	MOVQ  dst+0(FP), AX
	MOVQ  dst_len+8(FP), BX
	MOVSD s+24(FP), X15
	SHUFPD $0, X15, X15
	CMPQ BX, $2
	JL   done_msv2
loop_msv2:
	MOVUPD (AX), X0
	MULPD  X15, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_msv2
done_msv2:
	RET

// reduceSumSqSSE2(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceSumSqSSE2(SB), NOSPLIT, $0-32
	MOVQ  a+0(FP), AX
	MOVQ  a_len+8(FP), BX
	XORPD X0, X0
	CMPQ  BX, $2
	JL    done_rss2
loop_rss2:
	MOVUPD (AX), X1
	MULPD  X1, X1
	ADDPD  X1, X0
	ADDQ $16, AX
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_rss2
done_rss2:
	MOVAPD X0, X1
	UNPCKHPD X0, X1
	ADDSD X1, X0
	MOVSD  X0, ret+24(FP)
	RET
