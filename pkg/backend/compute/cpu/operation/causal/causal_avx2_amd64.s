#include "textflag.h"

// Causal linear-algebra kernels: AVX2/FMA implementations live in this file.
//
// Several symbols are named *SSE2 (e.g. ·matVecSSE2) but are defined here alongside
// AVX2 variants so one translation unit satisfies both code paths selected at runtime
// in causal_amd64.go. The sibling causal_sse2_amd64.s file contains only comments to
// document that split and avoid duplicate TEXT symbols at link time.

// matVecAVX2(dst, w, x []float64, rows, cols int)
// dst = W @ x  where W is [rows x cols] row-major.
// ABI0: dst+0(FP)..16, w+24(FP)..40, x+48(FP)..64, rows+72(FP), cols+80(FP)
TEXT ·matVecAVX2(SB), NOSPLIT, $0-88
	MOVQ dst+0(FP), AX        // dst pointer
	MOVQ w+24(FP), BX         // W pointer
	MOVQ x+48(FP), CX         // x pointer
	MOVQ rows+72(FP), DX      // rows
	MOVQ cols+80(FP), SI      // cols
	TESTQ DX, DX
	JZ   done_mv_avx2
row_loop_mv_avx2:
	VXORPD Y0, Y0, Y0          // accumulator = 0
	VXORPD Y5, Y5, Y5          // accumulator 2 = 0
	MOVQ   SI, DI              // cols remaining
	MOVQ   CX, R8              // x pointer (reset per row)
	CMPQ   DI, $8
	JL     try_4_mv_avx2
col_loop_8_mv_avx2:
	VMOVUPD (BX), Y1
	VMOVUPD (R8), Y2
	VFMADD231PD Y2, Y1, Y0
	VMOVUPD 32(BX), Y3
	VMOVUPD 32(R8), Y4
	VFMADD231PD Y4, Y3, Y5
	ADDQ $64, BX
	ADDQ $64, R8
	SUBQ $8, DI
	CMPQ DI, $8
	JGE  col_loop_8_mv_avx2
try_4_mv_avx2:
	CMPQ   DI, $4
	JL     tail_mv_avx2
col_loop_mv_avx2:
	VMOVUPD (BX), Y1
	VMOVUPD (R8), Y2
	VFMADD231PD Y2, Y1, Y0
	ADDQ $32, BX
	ADDQ $32, R8
	SUBQ $4, DI
	CMPQ DI, $4
	JGE  col_loop_mv_avx2
tail_mv_avx2:
	VADDPD Y5, Y0, Y0
	// Horizontal sum of Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
	// Handle remaining cols
	TESTQ DI, DI
	JZ    store_mv_avx2
scalar_mv_avx2:
	VMOVSD (BX), X1
	VMOVSD (R8), X2
	VFMADD231SD X2, X1, X0
	ADDQ $8, BX
	ADDQ $8, R8
	DECQ DI
	JNZ  scalar_mv_avx2
store_mv_avx2:
	VMOVSD X0, (AX)
	ADDQ $8, AX
	DECQ DX
	JNZ  row_loop_mv_avx2
done_mv_avx2:
	VZEROUPPER
	RET

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

// axpyAVX2(dst, src []float64, scale float64)
// dst[i] += scale * src[i]
// ABI0: dst+0(FP)..16, src+24(FP)..40, scale+48(FP)
TEXT ·axpyAVX2(SB), NOSPLIT, $0-56
	MOVQ         dst+0(FP), AX
	MOVQ         src_len+32(FP), BX
	MOVQ         src+24(FP), DI
	VBROADCASTSD scale+48(FP), Y15
	CMPQ BX, $8
	JL   try_4_axpy_avx2
loop_8_axpy_avx2:
	VMOVUPD (AX), Y0
	VMOVUPD (DI), Y1
	VFMADD231PD Y15, Y1, Y0
	VMOVUPD Y0, (AX)
	VMOVUPD 32(AX), Y2
	VMOVUPD 32(DI), Y3
	VFMADD231PD Y15, Y3, Y2
	VMOVUPD Y2, 32(AX)
	ADDQ $64, AX
	ADDQ $64, DI
	SUBQ $8, BX
	CMPQ BX, $8
	JGE  loop_8_axpy_avx2
try_4_axpy_avx2:
	CMPQ BX, $4
	JL   tail_axpy_avx2
loop_axpy_avx2:
	VMOVUPD (AX), Y0
	VMOVUPD (DI), Y1
	VFMADD231PD Y15, Y1, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_axpy_avx2
tail_axpy_avx2:
	TESTQ BX, BX
	JZ    done_axpy_avx2
	MOVSD scale+48(FP), X14
scalar_axpy_avx2:
	VMOVSD (AX), X0
	VMOVSD (DI), X1
	VFMADD231SD X0, X1, X14
	VMOVSD X0, (AX)
	ADDQ $8, AX
	ADDQ $8, DI
	DECQ BX
	JNZ  scalar_axpy_avx2
done_axpy_avx2:
	VZEROUPPER
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

// dotAVX2(a, b []float64) float64
// ABI0: a+0..16, b+24..40, ret+48
TEXT ·dotAVX2(SB), NOSPLIT, $0-56
	MOVQ a+0(FP), AX
	MOVQ a_len+8(FP), BX
	MOVQ b+24(FP), DI
	VXORPD Y0, Y0, Y0
	VXORPD Y5, Y5, Y5
	CMPQ BX, $8
	JL   try_4_dot_avx2
loop_8_dot_avx2:
	VMOVUPD (AX), Y1
	VMOVUPD (DI), Y2
	VFMADD231PD Y2, Y1, Y0
	VMOVUPD 32(AX), Y3
	VMOVUPD 32(DI), Y4
	VFMADD231PD Y4, Y3, Y5
	ADDQ $64, AX
	ADDQ $64, DI
	SUBQ $8, BX
	CMPQ BX, $8
	JGE  loop_8_dot_avx2
try_4_dot_avx2:
	CMPQ BX, $4
	JL   tail_dot_avx2
loop_dot_avx2:
	VMOVUPD (AX), Y1
	VMOVUPD (DI), Y2
	VFMADD231PD Y2, Y1, Y0
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_dot_avx2
tail_dot_avx2:
	VADDPD Y5, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
	TESTQ BX, BX
	JZ    done_dot_avx2
scalar_dot_avx2:
	VMOVSD (AX), X1
	VMOVSD (DI), X2
	VFMADD231SD X2, X1, X0
	ADDQ $8, AX
	ADDQ $8, DI
	DECQ BX
	JNZ  scalar_dot_avx2
done_dot_avx2:
	VMOVSD X0, ret+48(FP)
	VZEROUPPER
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

// subVecAVX2(dst, a, b []float64)
// dst[i] = a[i] - b[i]
// ABI0: dst+0..16, a+24..40, b+48..64
TEXT ·subVecAVX2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ a_len+32(FP), BX
	MOVQ a+24(FP), DI
	MOVQ b+48(FP), SI
	CMPQ BX, $4
	JL   tail_sub_avx2
loop_sub_avx2:
	VMOVUPD (DI), Y0
	VMOVUPD (SI), Y1
	VSUBPD  Y1, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_sub_avx2
tail_sub_avx2:
	TESTQ BX, BX
	JZ    done_sub_avx2
scalar_sub_avx2:
	VMOVSD (DI), X0
	VMOVSD (SI), X1
	VSUBSD  X1, X0, X0
	VMOVSD X0, (AX)
	ADDQ $8, AX
	ADDQ $8, DI
	ADDQ $8, SI
	DECQ BX
	JNZ  scalar_sub_avx2
done_sub_avx2:
	VZEROUPPER
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
