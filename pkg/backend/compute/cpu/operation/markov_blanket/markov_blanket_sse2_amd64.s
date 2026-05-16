#include "textflag.h"

// matvecSSE2(dst, w, x []float64, rows, cols int)
TEXT ·matvecSSE2(SB), NOSPLIT, $0-88
	MOVQ dst+0(FP), AX
	MOVQ w+24(FP), BX
	MOVQ x+48(FP), DX
	MOVQ rows+72(FP), R8
	MOVQ cols+80(FP), R9
	TESTQ R8, R8
	JZ   done_mvs
row_loop_mvs:
	XORPD  X0, X0
	MOVQ   R9, CX
	MOVQ   BX, SI
	MOVQ   DX, DI
	CMPQ   CX, $4
	JL     tail_mvs
vec_loop_4_mvs:
	MOVUPD (SI), X1
	MOVUPD (DI), X2
	MULPD  X2, X1
	ADDPD  X1, X0
	MOVUPD 16(SI), X1
	MOVUPD 16(DI), X2
	MULPD  X2, X1
	ADDPD  X1, X0
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $4, CX
	CMPQ CX, $4
	JGE  vec_loop_4_mvs
	CMPQ CX, $2
	JLT  tail_mvs
vec_loop_mvs:
	MOVUPD (SI), X1
	MOVUPD (DI), X2
	MULPD  X2, X1
	ADDPD  X1, X0
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $2, CX
	CMPQ CX, $2
	JGE  vec_loop_mvs
tail_mvs:
	HADDPD X0, X0
scalar_rem_mvs:
	TESTQ  CX, CX
	JZ     write_mvs
	MOVSD  (SI), X1
	MOVSD  (DI), X2
	MULSD  X2, X1
	ADDSD  X1, X0
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ  scalar_rem_mvs
write_mvs:
	MOVSD  (AX), X3
	ADDSD  X0, X3
	MOVSD  X3, (AX)
	ADDQ $8, AX
	MOVQ R9, CX
	SHLQ $3, CX
	ADDQ CX, BX
	DECQ R8
	JNZ  row_loop_mvs
done_mvs:
	RET

// subVecSSE2(dst, a, b []float64)  dst[i] = a[i] - b[i]
TEXT ·subVecSSE2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ dst_len+8(FP), BX
	MOVQ a_len+32(FP), CX
	CMPQ CX, BX
	JAE  mb_sv2_min1
	MOVQ CX, BX
mb_sv2_min1:
	MOVQ b_len+56(FP), CX
	CMPQ CX, BX
	JAE  mb_sv2_min2
	MOVQ CX, BX
mb_sv2_min2:
	MOVQ a+24(FP), DI
	MOVQ b+48(FP), SI
	CMPQ BX, $2
	JL   tail_mb_sv2
loop_sv_sse:
	MOVUPD (DI), X0
	MOVUPD (SI), X1
	SUBPD  X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_sv_sse
tail_mb_sv2:
	CMPQ BX, $1
	JNE  done_sv_sse
	MOVSD (DI), X0
	MOVSD (SI), X1
	SUBSD X1, X0
	MOVSD X0, (AX)
done_sv_sse:
	RET
