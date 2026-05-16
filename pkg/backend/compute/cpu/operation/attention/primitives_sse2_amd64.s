#include "textflag.h"

// dotProductSSE2(a, b []float64) float64
// ABI0: same layout, ret+48(FP)
TEXT ·dotProductSSE2(SB), NOSPLIT, $0-56
	MOVQ  a+0(FP), AX
	MOVQ  a_len+8(FP), BX
	MOVQ  b+24(FP), DI
	XORPS X0, X0
	CMPQ  BX, $2
	JL    scalar_dp2
loop_dp2:
	MOVUPD (AX), X1
	MOVUPD (DI), X2
	MULPD  X2, X1
	ADDPD  X1, X0
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_dp2
scalar_dp2:
	HADDPD X0, X0
	CMPQ BX, $0
	JLE  done_dp2
tail_dp2:
	MOVSD (AX), X1
	MULSD (DI), X1
	ADDSD X1, X0
	ADDQ $8, AX
	ADDQ $8, DI
	DECQ BX
	JNZ  tail_dp2
done_dp2:
	MOVSD  X0, ret+48(FP)
	RET

// scaledAddSSE2(dst, src []float64, scale float64)
TEXT ·scaledAddSSE2(SB), NOSPLIT, $0-56
	MOVQ   dst+0(FP), AX
	MOVQ   src_len+32(FP), BX
	MOVQ   src+24(FP), DI
	MOVSD  scale+48(FP), X15
	UNPCKLPD X15, X15
	CMPQ   BX, $2
	JL     tail_sa2
loop_sa2:
	MOVUPD (DI), X1
	MOVUPD (AX), X0
	MULPD  X15, X1
	ADDPD  X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_sa2
tail_sa2:
	CMPQ BX, $0
	JLE  done_sa2
	MOVSD (DI), X1
	MULSD X15, X1
	MOVSD (AX), X0
	ADDSD X1, X0
	MOVSD X0, (AX)
done_sa2:
	RET

// reduceMaxSSE2(a []float64) float64
TEXT ·reduceMaxSSE2(SB), NOSPLIT, $0-32
	MOVQ   a+0(FP), AX
	MOVQ   a_len+8(FP), BX
	CMPQ   BX, $2
	JL     scalar_rm2_init
	MOVUPD (AX), X0
	ADDQ $16, AX
	SUBQ $2, BX
loop_rm2:
	MOVUPD (AX), X1
	MAXPD  X1, X0
	ADDQ $16, AX
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_rm2
scalar_rm2:
	MOVUPD X0, X1
	UNPCKHPD X0, X1
	MAXSD  X1, X0
	JMP    tail_rm2
scalar_rm2_init:
	MOVSD (AX), X0
	ADDQ  $8, AX
	DECQ  BX
tail_rm2:
	CMPQ BX, $0
	JLE  done_rm2
tail_rm2_loop:
	MOVSD (AX), X1
	MAXSD X1, X0
	ADDQ $8, AX
	DECQ BX
	JNZ  tail_rm2_loop
done_rm2:
	MOVSD  X0, ret+24(FP)
	RET

// reduceSumSSE2(a []float64) float64
TEXT ·reduceSumSSE2(SB), NOSPLIT, $0-32
	MOVQ   a+0(FP), AX
	MOVQ   a_len+8(FP), BX
	XORPS  X0, X0
	CMPQ   BX, $2
	JL     done_rs2
loop_rs2:
	MOVUPD (AX), X1
	ADDPD  X1, X0
	ADDQ $16, AX
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_rs2
done_rs2:
	HADDPD X0, X0
	CMPQ BX, $0
	JLE  done_rs2_ret
tail_rs2:
	ADDSD (AX), X0
	ADDQ $8, AX
	DECQ BX
	JNZ  tail_rs2
done_rs2_ret:
	MOVSD  X0, ret+24(FP)
	RET

// divScalarSSE2(dst []float64, s float64)
TEXT ·divScalarSSE2(SB), NOSPLIT, $0-32
	MOVQ   dst+0(FP), AX
	MOVQ   dst_len+8(FP), BX
	MOVSD  s+24(FP), X15
	UNPCKLPD X15, X15
	CMPQ   BX, $2
	JL     tail_ds2
loop_ds2:
	MOVUPD (AX), X0
	DIVPD  X15, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_ds2
tail_ds2:
	CMPQ BX, $0
	JLE  done_ds2
	MOVSD (AX), X0
	DIVSD X15, X0
	MOVSD X0, (AX)
done_ds2:
	RET
