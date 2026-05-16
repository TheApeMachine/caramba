#include "textflag.h"

// matvecAVX2(dst, w, x []float64, rows, cols int)
// dst[i] += sum_j( w[i*cols+j] * x[j] )  — accumulates into dst (caller pre-fills with bias)
// ABI0 layout:
//   dst+0(FP)  ptr, +8 len, +16 cap
//   w+24(FP)   ptr, +32 len, +40 cap
//   x+48(FP)   ptr, +56 len, +64 cap
//   rows+72(FP)
//   cols+80(FP)
// matvecAVX2: 8-wide unrolled SIMD when cols≥8, then 4-wide, then scalar tail.
TEXT ·matvecAVX2(SB), NOSPLIT, $0-88
	MOVQ dst+0(FP), AX      // dst ptr
	MOVQ w+24(FP), BX       // w ptr
	MOVQ x+48(FP), DX       // x ptr
	MOVQ rows+72(FP), R8
	MOVQ cols+80(FP), R9
	TESTQ R8, R8
	JZ   done_mv2
row_loop_mv2:
	VXORPD Y0, Y0, Y0
	VXORPD Y5, Y5, Y5
	MOVQ   R9, CX           // col counter
	MOVQ   BX, SI           // row start in w
	MOVQ   DX, DI           // x ptr (reset each row)
	CMPQ   CX, $8
	JL     vec_try_4
vec_loop_8:
	VMOVUPD (SI), Y1
	VMOVUPD (DI), Y2
	VFMADD231PD Y2, Y1, Y0
	VMOVUPD 32(SI), Y3
	VMOVUPD 32(DI), Y4
	VFMADD231PD Y4, Y3, Y5
	ADDQ $64, SI
	ADDQ $64, DI
	SUBQ $8, CX
	CMPQ CX, $8
	JGE  vec_loop_8
vec_try_4:
	CMPQ CX, $4
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
	VADDPD Y5, Y0, Y0
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

// subVecAVX2(dst, a, b []float64)  dst[i] = a[i] - b[i]
// Precondition: min(len(dst),len(a),len(b))) elements are valid to write/read.
TEXT ·subVecAVX2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ dst_len+8(FP), BX
	MOVQ a_len+32(FP), CX
	CMPQ CX, BX
	JAE  mb_sv_min1
	MOVQ CX, BX
mb_sv_min1:
	MOVQ b_len+56(FP), CX
	CMPQ CX, BX
	JAE  mb_sv_min2
	MOVQ CX, BX
mb_sv_min2:
	MOVQ a+24(FP), DI
	MOVQ b+48(FP), SI
	CMPQ BX, $4
	JL   tail_mb_sv_avx
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
tail_mb_sv_avx:
	TESTQ BX, BX
	JZ    done_sv_avx
tail_mb_one_avx:
	MOVSD (DI), X0
	MOVSD (SI), X1
	SUBSD X1, X0
	MOVSD X0, (AX)
	ADDQ $8, AX
	ADDQ $8, DI
	ADDQ $8, SI
	DECQ BX
	JNZ   tail_mb_one_avx
done_sv_avx:
	VZEROUPPER
	RET
