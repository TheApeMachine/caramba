#include "textflag.h"

// matvecAVX2(dst, w, x []float64, rows, cols int)
// dst[i] += sum_j( w[i*cols+j] * x[j] )  — accumulates into dst (caller pre-fills with bias)
// ABI0 layout:
//   dst+0(FP)  ptr, +8 len, +16 cap
//   w+24(FP)   ptr, +32 len, +40 cap
//   x+48(FP)   ptr, +56 len, +64 cap
//   rows+72(FP)
//   cols+80(FP)
TEXT ·matvecAVX2(SB), NOSPLIT, $0-88
	MOVQ dst+0(FP), AX      // dst ptr
	MOVQ w+24(FP), BX       // w ptr
	MOVQ x+48(FP), DX       // x ptr
	MOVQ rows+72(FP), R8
	MOVQ cols+80(FP), R9
	TESTQ R8, R8
	JZ   done_mv2
row_loop_mv2:
	// accumulate dot product of w[row*cols:] with x into Y0
	VXORPD Y0, Y0, Y0
	MOVQ   R9, CX           // col counter
	MOVQ   BX, SI           // row start in w
	MOVQ   DX, DI           // x ptr (reset each row)
	CMPQ   CX, $4
	JL     tail_mv2
vec_loop_mv2:
	VMOVUPD (SI), Y1
	VMOVUPD (DI), Y2
	VFMADD231PD Y2, Y1, Y0
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $4, CX
	CMPQ CX, $4
	JGE  vec_loop_mv2
tail_mv2:
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
	// scalar tail
	TESTQ CX, CX
	JZ    write_mv2
scalar_mv2:
	VMOVSD (SI), X1
	VMOVSD (DI), X2
	VMULSD X2, X1, X1
	VADDSD X1, X0, X0
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNZ  scalar_mv2
write_mv2:
	// dst[row] += acc (X0)
	VMOVSD (AX), X3
	VADDSD X0, X3, X3
	VMOVSD X3, (AX)
	ADDQ $8, AX
	// advance w to next row
	MOVQ R9, CX
	SHLQ $3, CX
	ADDQ CX, BX
	DECQ R8
	JNZ  row_loop_mv2
done_mv2:
	VZEROUPPER
	RET

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
	XORPS  X0, X0
	MOVQ   R9, CX
	MOVQ   BX, SI
	MOVQ   DX, DI
	CMPQ   CX, $2
	JL     tail_mvs
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
	TESTQ  CX, CX
	JZ     write_mvs
	MOVSD  (SI), X1
	MOVSD  (DI), X2
	MULSD  X2, X1
	ADDSD  X1, X0
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

// subVecAVX2(dst, a, b []float64)  dst[i] = a[i] - b[i]
// ABI0: dst+0(FP)..16, a+24(FP)..40, b+48(FP)..64
TEXT ·subVecAVX2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ a_len+32(FP), BX
	MOVQ a+24(FP), DI
	MOVQ b+48(FP), SI
	CMPQ BX, $4
	JL   done_sv_avx
loop_sv_avx:
	VMOVUPD (DI), Y0
	VMOVUPD (SI), Y1
	VSUBPD  Y1, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_sv_avx
done_sv_avx:
	VZEROUPPER
	RET

// subVecSSE2(dst, a, b []float64)  dst[i] = a[i] - b[i]
TEXT ·subVecSSE2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ a_len+32(FP), BX
	MOVQ a+24(FP), DI
	MOVQ b+48(FP), SI
	CMPQ BX, $2
	JL   done_sv_sse
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
done_sv_sse:
	RET
