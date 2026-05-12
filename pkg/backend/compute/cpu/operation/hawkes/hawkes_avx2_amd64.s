#include "textflag.h"

// expSumAVX2(expBuf []float64) float64
TEXT ·expSumAVX2(SB), NOSPLIT, $0-32
	MOVQ   expBuf+0(FP), AX
	MOVQ   expBuf_len+8(FP), BX
	VXORPD Y0, Y0, Y0
	CMPQ   BX, $4
	JL     done_es_avx
loop_es_avx:
	VMOVUPD (AX), Y1
	VADDPD  Y1, Y0, Y0
	ADDQ $32, AX
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_es_avx
done_es_avx:
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
	TESTQ BX, BX
	JZ    end_es_avx
scalar_es_avx:
	VMOVSD (AX), X1
	VADDSD X1, X0, X0
	ADDQ $8, AX
	DECQ BX
	JNZ  scalar_es_avx
end_es_avx:
	MOVSD X0, ret+24(FP)
	VZEROUPPER
	RET

// expSumSSE2(expBuf []float64) float64
TEXT ·expSumSSE2(SB), NOSPLIT, $0-32
	MOVQ   expBuf+0(FP), AX
	MOVQ   expBuf_len+8(FP), BX
	XORPS  X0, X0
	CMPQ   BX, $2
	JL     done_es_sse
loop_es_sse:
	MOVUPD (AX), X1
	ADDPD  X1, X0
	ADDQ $16, AX
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_es_sse
done_es_sse:
	HADDPD X0, X0
	TESTQ BX, BX
	JZ    end_es_sse
scalar_es_sse:
	MOVSD  (AX), X1
	ADDSD  X1, X0
	ADDQ $8, AX
	DECQ BX
	JNZ  scalar_es_sse
end_es_sse:
	MOVSD X0, ret+24(FP)
	RET

// subVecAVX2(dst, a, b []float64)  dst[i] = a[i] - b[i]
// Precondition: caller must pass slices whose lengths are all >= the computed
// iteration count; we use n = min(len(dst), len(a), len(b)).
TEXT ·subVecAVX2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ dst_len+8(FP), BX
	MOVQ a_len+32(FP), CX
	CMPQ CX, BX
	JAE  sub_sv_min1
	MOVQ CX, BX
sub_sv_min1:
	MOVQ b_len+56(FP), CX
	CMPQ CX, BX
	JAE  sub_sv_min2
	MOVQ CX, BX
sub_sv_min2:
	MOVQ a+24(FP), DI
	MOVQ b+48(FP), SI
	CMPQ BX, $4
	JL   tail_sv_avx
loop_sv:
	VMOVUPD (DI), Y0
	VMOVUPD (SI), Y1
	VSUBPD  Y1, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_sv
tail_sv_avx:
	TESTQ BX, BX
	JZ    done_sv_avx
tail_sv_one:
	MOVSD (DI), X0
	MOVSD (SI), X1
	SUBSD X1, X0
	MOVSD X0, (AX)
	ADDQ $8, AX
	ADDQ $8, DI
	ADDQ $8, SI
	DECQ BX
	JNZ   tail_sv_one
done_sv_avx:
	VZEROUPPER
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
	ADDQ $16, AX
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_sv2
tail_sv2:
	CMPQ BX, $1
	JNE  done_sv2
	MOVSD (DI), X0
	MOVSD (SI), X1
	SUBSD X1, X0
	MOVSD X0, (AX)
done_sv2:
	RET
