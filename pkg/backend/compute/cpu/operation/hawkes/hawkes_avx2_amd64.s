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
