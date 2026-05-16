#include "textflag.h"

// matVecSSE2(dst, w, x []float64, rows, cols int)
TEXT ·matVecSSE2(SB), NOSPLIT, $0-88
	MOVQ dst+0(FP), AX
	MOVQ w+24(FP), BX
	MOVQ x+48(FP), CX
	MOVQ rows+72(FP), DX
	MOVQ cols+80(FP), SI
	TESTQ DX, DX
	JZ   done_mv_sse2
row_loop_mv_sse2:
	XORPD  X0, X0
	MOVQ   SI, DI
	MOVQ   CX, R8
	CMPQ   DI, $2
	JL     tail_mv_sse2
col_loop_mv_sse2:
	MOVUPD (BX), X1
	MOVUPD (R8), X2
	MULPD  X2, X1
	ADDPD  X1, X0
	ADDQ $16, BX
	ADDQ $16, R8
	SUBQ $2, DI
	CMPQ DI, $2
	JGE  col_loop_mv_sse2
tail_mv_sse2:
	HADDPD X0, X0
	TESTQ DI, DI
	JZ    store_mv_sse2
scalar_mv_sse2:
	MOVSD  (BX), X1
	MOVSD  (R8), X2
	MULSD  X2, X1
	ADDSD  X1, X0
	ADDQ $8, BX
	ADDQ $8, R8
	DECQ DI
	JNZ  scalar_mv_sse2
store_mv_sse2:
	MOVSD  X0, (AX)
	ADDQ $8, AX
	DECQ DX
	JNZ  row_loop_mv_sse2
done_mv_sse2:
	RET

// axpySSE2(dst, src []float64, scale float64)
TEXT ·axpySSE2(SB), NOSPLIT, $0-56
	MOVQ  dst+0(FP), AX
	MOVQ  src_len+32(FP), BX
	MOVQ  src+24(FP), DI
	MOVSD scale+48(FP), X15
	SHUFPD $0, X15, X15
	CMPQ BX, $2
	JL   tail_axpy_sse2
loop_axpy_sse2:
	MOVUPD (AX), X0
	MOVUPD (DI), X1
	MULPD  X15, X1
	ADDPD  X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_axpy_sse2
tail_axpy_sse2:
	TESTQ BX, BX
	JZ    done_axpy_sse2
	MOVSD scale+48(FP), X14
scalar_axpy_sse2:
	MOVSD (AX), X0
	MOVSD (DI), X1
	MULSD  X14, X1
	ADDSD  X1, X0
	MOVSD  X0, (AX)
	ADDQ $8, AX
	ADDQ $8, DI
	DECQ BX
	JNZ  scalar_axpy_sse2
done_axpy_sse2:
	RET

// dotSSE2(a, b []float64) float64
// ABI0: a+0..16, b+24..40, ret+48
TEXT ·dotSSE2(SB), NOSPLIT, $0-56
	MOVQ a+0(FP), AX
	MOVQ a_len+8(FP), BX
	MOVQ b+24(FP), DI
	XORPS X0, X0
	CMPQ BX, $2
	JL   tail_dot_sse2
loop_dot_sse2:
	MOVUPD (AX), X1
	MOVUPD (DI), X2
	MULPD  X2, X1
	ADDPD  X1, X0
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_dot_sse2
tail_dot_sse2:
	HADDPD X0, X0
	TESTQ BX, BX
	JZ    done_dot_sse2
scalar_dot_sse2:
	MOVSD  (AX), X1
	MOVSD  (DI), X2
	MULSD  X2, X1
	ADDSD  X1, X0
	ADDQ $8, AX
	ADDQ $8, DI
	DECQ BX
	JNZ  scalar_dot_sse2
done_dot_sse2:
	MOVSD X0, ret+48(FP)
	RET

// subVecSSE2(dst, a, b []float64)
TEXT ·subVecSSE2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ a_len+32(FP), BX
	MOVQ a+24(FP), DI
	MOVQ b+48(FP), SI
	CMPQ BX, $2
	JL   tail_sub_sse2
loop_sub_sse2:
	MOVUPD (DI), X0
	MOVUPD (SI), X1
	SUBPD  X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_sub_sse2
tail_sub_sse2:
	CMPQ BX, $1
	JNE  done_sub_sse2
	MOVSD (DI), X0
	MOVSD (SI), X1
	SUBSD X1, X0
	MOVSD X0, (AX)
done_sub_sse2:
	RET
