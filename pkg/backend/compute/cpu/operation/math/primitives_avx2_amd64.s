#include "textflag.h"

// reduceSumAVX2(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceSumAVX2(SB), NOSPLIT, $0-32
	MOVQ   a+0(FP), AX
	MOVQ   a_len+8(FP), BX
	VXORPD Y0, Y0, Y0
	VXORPD Y5, Y5, Y5
	CMPQ   BX, $8
	JL     try_4_rs
loop_8_rs:
	VMOVUPD (AX), Y1
	VADDPD  Y1, Y0, Y0
	VMOVUPD 32(AX), Y2
	VADDPD  Y2, Y5, Y5
	ADDQ $64, AX
	SUBQ $8, BX
	CMPQ BX, $8
	JGE  loop_8_rs
try_4_rs:
	CMPQ   BX, $4
	JL     done_rs
loop_rs:
	VMOVUPD (AX), Y1
	VADDPD  Y1, Y0, Y0
	ADDQ $32, AX
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_rs
done_rs:
	VADDPD Y5, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
	MOVSD  X0, ret+24(FP)
	VZEROUPPER
	RET

// reduceMaxAVX2(a []float64) float64
TEXT ·reduceMaxAVX2(SB), NOSPLIT, $0-32
	MOVQ    a+0(FP), AX
	MOVQ    a_len+8(FP), BX
	CMPQ    BX, $4
	JL      scalar_rm
	VMOVUPD (AX), Y0
	ADDQ $32, AX
	SUBQ $4, BX
loop_rm:
	VMOVUPD (AX), Y1
	VMAXPD  Y1, Y0, Y0
	ADDQ $32, AX
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_rm
scalar_rm:
	VEXTRACTF128 $1, Y0, X1
	VMAXPD X1, X0, X0
	VUNPCKHPD X0, X0, X1
	VMAXSD X1, X0, X0
	MOVSD  X0, ret+24(FP)
	VZEROUPPER
	RET

// divScalarAVX2(dst []float64, s float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP), s+24(FP)
TEXT ·divScalarAVX2(SB), NOSPLIT, $0-32
	MOVQ     dst+0(FP), AX
	MOVQ     dst_len+8(FP), BX
	VMOVSD   s+24(FP), X15
	VBROADCASTSD X15, Y15
	CMPQ     BX, $4
	JL       done_ds
loop_ds:
	VMOVUPD (AX), Y0
	VDIVPD  Y15, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_ds
done_ds:
	VZEROUPPER
	RET

// addVecAVX2(dst, a, b []float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP),
//       a+24(FP), a_len+32(FP), a_cap+40(FP),
//       b+48(FP), b_len+56(FP), b_cap+64(FP)
TEXT ·addVecAVX2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ a_len+32(FP), BX
	MOVQ a+24(FP), DI
	MOVQ b+48(FP), SI
	CMPQ BX, $4
	JL   done_av
loop_av:
	VMOVUPD (DI), Y0
	VMOVUPD (SI), Y1
	VADDPD  Y1, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_av
done_av:
	VZEROUPPER
	RET

// mulVecAVX2(dst, a, b []float64)
TEXT ·mulVecAVX2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ a_len+32(FP), BX
	MOVQ a+24(FP), DI
	MOVQ b+48(FP), SI
	CMPQ BX, $4
	JL   done_mv
loop_mv:
	VMOVUPD (DI), Y0
	VMOVUPD (SI), Y1
	VMULPD  Y1, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_mv
done_mv:
	VZEROUPPER
	RET

// mulScalarAVX2(dst []float64, s float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP), s+24(FP)
TEXT ·mulScalarAVX2(SB), NOSPLIT, $0-32
	MOVQ     dst+0(FP), AX
	MOVQ     dst_len+8(FP), BX
	VMOVSD   s+24(FP), X15
	VBROADCASTSD X15, Y15
	CMPQ     BX, $4
	JL       done_ms
loop_ms:
	VMOVUPD (AX), Y0
	VMULPD  Y15, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_ms
done_ms:
	VZEROUPPER
	RET

// reduceSumSqAVX2(a []float64) float64
// Computes sum of squares: sum(a[i]*a[i])
TEXT ·reduceSumSqAVX2(SB), NOSPLIT, $0-32
	MOVQ   a+0(FP), AX
	MOVQ   a_len+8(FP), BX
	VXORPD Y0, Y0, Y0
	VXORPD Y5, Y5, Y5
	CMPQ   BX, $8
	JL     try_4_ssq
loop_8_ssq:
	VMOVUPD (AX), Y1
	VFMADD231PD Y1, Y1, Y0
	VMOVUPD 32(AX), Y2
	VFMADD231PD Y2, Y2, Y5
	ADDQ $64, AX
	SUBQ $8, BX
	CMPQ BX, $8
	JGE  loop_8_ssq
try_4_ssq:
	CMPQ   BX, $4
	JL     done_ssq
loop_ssq:
	VMOVUPD (AX), Y1
	VFMADD231PD Y1, Y1, Y0
	ADDQ $32, AX
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_ssq
done_ssq:
	VADDPD Y5, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
	MOVSD  X0, ret+24(FP)
	VZEROUPPER
	RET

// signVecAVX2(dst, src []float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP),
//       src+24(FP), src_len+32(FP), src_cap+40(FP)
// dst[i] = +1 if src[i] > 0, -1 if src[i] < 0, 0 if src[i] == 0
// Strategy: cmp > 0 → mask1, cmp < 0 → mask2, blend +1/-1/0 via masks.
TEXT ·signVecAVX2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), DI
	MOVQ src_len+32(FP), BX
	VXORPD Y1, Y1, Y1
	VMOVSD ·signOne_amd64(SB), X2
	VBROADCASTSD X2, Y2
	VMOVSD ·signNegOne_amd64(SB), X3
	VBROADCASTSD X3, Y3
	CMPQ BX, $4
	JL   done_sv_avx2
loop_sv_avx2:
	VMOVUPD (DI), Y0
	VCMPPD  $14, Y1, Y0, Y4         // Y4 = (Y0 > 0) mask
	VCMPPD  $1,  Y1, Y0, Y5         // Y5 = (Y0 < 0) mask
	VANDPD  Y2, Y4, Y4
	VANDPD  Y3, Y5, Y5
	VORPD   Y5, Y4, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_sv_avx2
done_sv_avx2:
	VZEROUPPER
	RET

// outerRowAVX2(dst, b []float64, scale float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP),
//       b+24(FP),  b_len+32(FP),  b_cap+40(FP),
//       scale+48(FP)
// dst[j] = scale * b[j]
TEXT ·outerRowAVX2(SB), NOSPLIT, $0-56
	MOVQ   dst+0(FP), AX
	MOVQ   b_len+32(FP), BX
	MOVQ   b+24(FP), DI
	VBROADCASTSD scale+48(FP), Y15
	CMPQ BX, $4
	JL   tail_or
loop_or:
	VMOVUPD (DI), Y0
	VMULPD  Y15, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_or
tail_or:
	VZEROUPPER
	RET

// addScaledVecAVX2(dst, src []float64, scale float64)
// dst[i] += scale * src[i]   — AXPY fused via VFMADD231PD
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP),
//       src+24(FP), src_len+32(FP), src_cap+40(FP),
//       scale+48(FP)
TEXT ·addScaledVecAVX2(SB), NOSPLIT, $0-56
	MOVQ   dst+0(FP), AX
	MOVQ   src_len+32(FP), BX
	MOVQ   src+24(FP), DI
	VBROADCASTSD scale+48(FP), Y15
	CMPQ BX, $4
	JL   done_asv
loop_asv:
	VMOVUPD (AX), Y0
	VMOVUPD (DI), Y1
	VFMADD231PD Y15, Y1, Y0      // Y0 += scale * Y1
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_asv
done_asv:
	VZEROUPPER
	RET

// sqrtVecAVX2(dst, src []float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP),
//       src+24(FP), src_len+32(FP), src_cap+40(FP)
TEXT ·sqrtVecAVX2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src_len+32(FP), BX
	MOVQ src+24(FP), DI
	CMPQ BX, $4
	JL   done_sv2
loop_sv2:
	VMOVUPD  (DI), Y0
	VSQRTPD  Y0, Y0
	VMOVUPD  Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_sv2
done_sv2:
	VZEROUPPER
	RET

// addScalarVecAVX2(dst []float64, scalar float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP), scalar+24(FP)
TEXT ·addScalarVecAVX2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), AX
	MOVQ dst_len+8(FP), BX
	VBROADCASTSD scalar+24(FP), Y15
	CMPQ BX, $4
	JL   done_asa
loop_asa:
	VMOVUPD (AX), Y0
	VADDPD  Y15, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_asa
done_asa:
	VZEROUPPER
	RET

// divVecAVX2(dst, a, b []float64)
// ABI0: dst+0(FP)..16, a+24(FP)..40, b+48(FP)..64
TEXT ·divVecAVX2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ a_len+32(FP), BX
	MOVQ a+24(FP), DI
	MOVQ b+48(FP), SI
	CMPQ BX, $4
	JL   done_dv
loop_dv:
	VMOVUPD (DI), Y0
	VMOVUPD (SI), Y1
	VDIVPD  Y1, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_dv
done_dv:
	VZEROUPPER
	RET

// clampVecAVX2(dst []float64, lo, hi float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP), lo+24(FP), hi+32(FP)
TEXT ·clampVecAVX2(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), AX
	MOVQ dst_len+8(FP), BX
	VBROADCASTSD lo+24(FP), Y14
	VBROADCASTSD hi+32(FP), Y15
	CMPQ BX, $4
	JL   done_cv
loop_cv:
	VMOVUPD (AX), Y0
	VMAXPD  Y14, Y0, Y0
	VMINPD  Y15, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_cv
done_cv:
	VZEROUPPER
	RET
