#include "textflag.h"

// matVecAVX2(dst, W, x []float64, rows, cols int)
// ABI0: dst+0(FP) dst_len+8(FP) dst_cap+16(FP)
//       W+24(FP)  W_len+32(FP)  W_cap+40(FP)
//       x+48(FP)  x_len+56(FP)  x_cap+64(FP)
//       rows+72(FP) cols+80(FP)
// Computes dst[i] = dot(W[i*cols:(i+1)*cols], x) for i in [0, rows).
TEXT ·matVecAVX2(SB), NOSPLIT, $0-88
	MOVQ dst+0(FP), DI
	MOVQ W+24(FP), SI
	MOVQ x+48(FP), DX
	MOVQ rows+72(FP), R8
	MOVQ cols+80(FP), R9
	XORQ R10, R10           // i = 0
row_loop_mv:
	CMPQ R10, R8
	JGE  done_mv
	VXORPD Y0, Y0, Y0       // acc = 0 — reset per row
	MOVQ   DX, AX           // AX = &x[0]
	MOVQ   R9, BX           // BX = cols remaining
	CMPQ   BX, $4
	JL     tail_mv
vec_mv:
	VMOVUPD (SI), Y1
	VMOVUPD (AX), Y2
	VMULPD  Y2, Y1, Y1
	VADDPD  Y1, Y0, Y0
	ADDQ $32, SI
	ADDQ $32, AX
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  vec_mv
tail_mv:
	// horizontal sum Y0 → scalar
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
	// scalar tail
	CMPQ BX, $0
	JLE  store_mv
scalar_mv:
	MOVSD  (SI), X1
	MOVSD  (AX), X2
	MULSD  X2, X1
	ADDSD  X1, X0
	ADDQ $8, SI
	ADDQ $8, AX
	SUBQ $1, BX
	JNZ  scalar_mv
store_mv:
	MOVSD X0, (DI)
	ADDQ $8, DI
	INCQ R10
	JMP  row_loop_mv
done_mv:
	VZEROUPPER
	RET

// matVecSSE2(dst, W, x []float64, rows, cols int)
TEXT ·matVecSSE2(SB), NOSPLIT, $0-88
	MOVQ dst+0(FP), DI
	MOVQ W+24(FP), SI
	MOVQ x+48(FP), DX
	MOVQ rows+72(FP), R8
	MOVQ cols+80(FP), R9
	XORQ R10, R10
row_loop_mv2:
	CMPQ R10, R8
	JGE  done_mv2
	XORPS X0, X0
	MOVQ  DX, AX
	MOVQ  R9, BX
	CMPQ  BX, $2
	JL    tail_mv2
vec_mv2:
	MOVUPD (SI), X1
	MOVUPD (AX), X2
	MULPD  X2, X1
	ADDPD  X1, X0
	ADDQ $16, SI
	ADDQ $16, AX
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  vec_mv2
tail_mv2:
	MOVUPD X0, X1
	UNPCKHPD X0, X1
	ADDSD  X1, X0
	CMPQ BX, $0
	JLE  store_mv2
	MOVSD (SI), X1
	MOVSD (AX), X2
	MULSD X2, X1
	ADDSD X1, X0
	ADDQ $8, SI
	ADDQ $8, AX
store_mv2:
	MOVSD X0, (DI)
	ADDQ $8, DI
	INCQ R10
	JMP  row_loop_mv2
done_mv2:
	RET

// matVecTransposeAVX2(dst, W, x []float64, rows, cols int)
// Computes dst[j] = sum_i W[i*cols+j] * x[i]  (W^T @ x)
// rows = num rows of W, cols = num cols of W → dst has length cols.
TEXT ·matVecTransposeAVX2(SB), NOSPLIT, $0-88
	MOVQ dst+0(FP), DI
	MOVQ W+24(FP), SI
	MOVQ x+48(FP), DX
	MOVQ rows+72(FP), R8
	MOVQ cols+80(FP), R9
	// zero dst
	MOVQ R9, BX
	MOVQ DI, AX
zero_loop:
	CMPQ BX, $0
	JLE  zero_done
	MOVQ $0, (AX)
	ADDQ $8, AX
	DECQ BX
	JMP  zero_loop
zero_done:
	// For each row i: dst[j] += W[i*cols+j] * x[i]
	XORQ R10, R10
outer_tv:
	CMPQ R10, R8
	JGE  done_tv
	MOVSD (DX), X15          // X15 = x[i]
	VBROADCASTSD X15, Y15   // Y15 = x[i] broadcast
	ADDQ $8, DX
	MOVQ DI, AX             // AX = &dst[0]
	MOVQ R9, BX
	CMPQ BX, $4
	JL   tail_tv
vec_tv:
	VMOVUPD (SI), Y1
	VMOVUPD (AX), Y2
	VMULPD  Y15, Y1, Y1
	VADDPD  Y1, Y2, Y2
	VMOVUPD Y2, (AX)
	ADDQ $32, SI
	ADDQ $32, AX
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  vec_tv
tail_tv:
	CMPQ BX, $0
	JLE  next_tv
scalar_tv:
	MOVSD (SI), X1
	MULSD X15, X1
	ADDSD (AX), X1
	MOVSD X1, (AX)
	ADDQ $8, SI
	ADDQ $8, AX
	DECQ BX
	JNZ  scalar_tv
next_tv:
	INCQ R10
	JMP  outer_tv
done_tv:
	VZEROUPPER
	RET

// matVecTransposeSSE2(dst, W, x []float64, rows, cols int)
TEXT ·matVecTransposeSSE2(SB), NOSPLIT, $0-88
	MOVQ dst+0(FP), DI
	MOVQ W+24(FP), SI
	MOVQ x+48(FP), DX
	MOVQ rows+72(FP), R8
	MOVQ cols+80(FP), R9
	MOVQ R9, BX
	MOVQ DI, AX
zero_loop2:
	CMPQ BX, $0
	JLE  zero_done2
	MOVQ $0, (AX)
	ADDQ $8, AX
	DECQ BX
	JMP  zero_loop2
zero_done2:
	XORQ R10, R10
outer_tv2:
	CMPQ R10, R8
	JGE  done_tv2
	MOVSD (DX), X15
	ADDQ $8, DX
	MOVQ DI, AX
	MOVQ R9, BX
	CMPQ BX, $2
	JL   tail_tv2
vec_tv2:
	MOVUPD (SI), X1
	MOVUPD (AX), X2
	MOVUPD X15, X14
	UNPCKLPD X14, X14
	MULPD  X14, X1
	ADDPD  X1, X2
	MOVUPD X2, (AX)
	ADDQ $16, SI
	ADDQ $16, AX
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  vec_tv2
tail_tv2:
	CMPQ BX, $0
	JLE  next_tv2
	MOVSD (SI), X1
	MULSD X15, X1
	ADDSD (AX), X1
	MOVSD X1, (AX)
	ADDQ $8, SI
	ADDQ $8, AX
next_tv2:
	INCQ R10
	JMP  outer_tv2
done_tv2:
	RET

// subVecAVX2(dst, a, b []float64)
TEXT ·subVecAVX2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), DI
	MOVQ a+24(FP), SI
	MOVQ b+48(FP), DX
	MOVQ a_len+32(FP), BX
	CMPQ BX, $4
	JL   done_sv
loop_sv:
	VMOVUPD (SI), Y0
	VMOVUPD (DX), Y1
	VSUBPD  Y1, Y0, Y0
	VMOVUPD Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DX
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_sv
done_sv:
	VZEROUPPER
	RET

// subVecSSE2(dst, a, b []float64)
TEXT ·subVecSSE2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), DI
	MOVQ a+24(FP), SI
	MOVQ b+48(FP), DX
	MOVQ a_len+32(FP), BX
	CMPQ BX, $2
	JL   done_sv2
loop_sv2:
	MOVUPD (SI), X0
	MOVUPD (DX), X1
	SUBPD  X1, X0
	MOVUPD X0, (DI)
	ADDQ $16, SI
	ADDQ $16, DX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_sv2
done_sv2:
	RET

// mulVecAVX2(dst, a, b []float64)
TEXT ·mulVecAVX2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), DI
	MOVQ a+24(FP), SI
	MOVQ b+48(FP), DX
	MOVQ a_len+32(FP), BX
	CMPQ BX, $4
	JL   done_mulv
loop_mulv:
	VMOVUPD (SI), Y0
	VMOVUPD (DX), Y1
	VMULPD  Y1, Y0, Y0
	VMOVUPD Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DX
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_mulv
done_mulv:
	VZEROUPPER
	RET

// mulVecSSE2(dst, a, b []float64)
TEXT ·mulVecSSE2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), DI
	MOVQ a+24(FP), SI
	MOVQ b+48(FP), DX
	MOVQ a_len+32(FP), BX
	CMPQ BX, $2
	JL   done_mulv2
loop_mulv2:
	MOVUPD (SI), X0
	MOVUPD (DX), X1
	MULPD  X1, X0
	MOVUPD X0, (DI)
	ADDQ $16, SI
	ADDQ $16, DX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_mulv2
done_mulv2:
	RET

// axpyAVX2(dst, src []float64, scale float64)
// dst[i] += scale * src[i]
TEXT ·axpyAVX2(SB), NOSPLIT, $0-56
	MOVQ   dst+0(FP), DI
	MOVQ   src+24(FP), SI
	MOVQ   src_len+32(FP), BX
	VMOVSD scale+48(FP), X15
	VBROADCASTSD X15, Y15
	CMPQ BX, $4
	JL   done_axpy
loop_axpy:
	VMOVUPD (SI), Y0      // Y0 = src
	VMULPD  Y15, Y0, Y0   // Y0 = scale * src
	VMOVUPD (DI), Y1      // Y1 = dst
	VADDPD  Y0, Y1, Y1    // Y1 = dst + scale*src
	VMOVUPD Y1, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_axpy
done_axpy:
	VZEROUPPER
	RET

// axpySSE2(dst, src []float64, scale float64)
TEXT ·axpySSE2(SB), NOSPLIT, $0-56
	MOVQ  dst+0(FP), DI
	MOVQ  src+24(FP), SI
	MOVQ  src_len+32(FP), BX
	MOVSD scale+48(FP), X15
	UNPCKLPD X15, X15
	CMPQ BX, $2
	JL   done_axpy2
loop_axpy2:
	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MULPD  X15, X0
	ADDPD  X0, X1
	MOVUPD X1, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_axpy2
done_axpy2:
	RET

// outerRowAVX2(dst, b []float64, scale float64)
// dst[j] += scale * b[j]
TEXT ·outerRowAVX2(SB), NOSPLIT, $0-56
	MOVQ   dst+0(FP), DI
	MOVQ   b+24(FP), SI
	MOVQ   b_len+32(FP), BX
	VMOVSD scale+48(FP), X15
	VBROADCASTSD X15, Y15
	CMPQ BX, $4
	JL   done_or
loop_or:
	VMOVUPD (SI), Y0
	VMULPD  Y15, Y0, Y0
	VMOVUPD (DI), Y1
	VADDPD  Y0, Y1, Y1
	VMOVUPD Y1, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_or
done_or:
	VZEROUPPER
	RET

// outerRowSSE2(dst, b []float64, scale float64)
TEXT ·outerRowSSE2(SB), NOSPLIT, $0-56
	MOVQ  dst+0(FP), DI
	MOVQ  b+24(FP), SI
	MOVQ  b_len+32(FP), BX
	MOVSD scale+48(FP), X15
	UNPCKLPD X15, X15
	CMPQ BX, $2
	JL   done_or2
loop_or2:
	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MULPD  X15, X0
	ADDPD  X0, X1
	MOVUPD X1, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_or2
done_or2:
	RET
