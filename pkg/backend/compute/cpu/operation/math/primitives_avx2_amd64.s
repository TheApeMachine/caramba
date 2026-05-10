#include "textflag.h"

// reduceSumAVX2(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceSumAVX2(SB), NOSPLIT, $0-32
	MOVQ   a+0(FP), AX
	MOVQ   a_len+8(FP), BX
	VXORPD Y0, Y0, Y0
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
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
	MOVSD  X0, ret+24(FP)
	VZEROUPPER
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
	MOVSD  X0, ret+24(FP)
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

// reduceMaxSSE2(a []float64) float64
TEXT ·reduceMaxSSE2(SB), NOSPLIT, $0-32
	MOVQ   a+0(FP), AX
	MOVQ   a_len+8(FP), BX
	CMPQ   BX, $2
	JL     scalar_rm2
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
	MOVSD  X0, ret+24(FP)
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

// divScalarSSE2(dst []float64, s float64)
TEXT ·divScalarSSE2(SB), NOSPLIT, $0-32
	MOVQ   dst+0(FP), AX
	MOVQ   dst_len+8(FP), BX
	MOVSD  s+24(FP), X15
	UNPCKLPD X15, X15
	CMPQ   BX, $2
	JL     done_ds2
loop_ds2:
	MOVUPD (AX), X0
	DIVPD  X15, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_ds2
done_ds2:
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

// addVecSSE2(dst, a, b []float64)
TEXT ·addVecSSE2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ a_len+32(FP), BX
	MOVQ a+24(FP), DI
	MOVQ b+48(FP), SI
	CMPQ BX, $2
	JL   done_av2
loop_av2:
	MOVUPD (DI), X0
	MOVUPD (SI), X1
	ADDPD  X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_av2
done_av2:
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

// mulVecSSE2(dst, a, b []float64)
TEXT ·mulVecSSE2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ a_len+32(FP), BX
	MOVQ a+24(FP), DI
	MOVQ b+48(FP), SI
	CMPQ BX, $2
	JL   done_mv2
loop_mv2:
	MOVUPD (DI), X0
	MOVUPD (SI), X1
	MULPD  X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_mv2
done_mv2:
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

// mulScalarSSE2(dst []float64, s float64)
TEXT ·mulScalarSSE2(SB), NOSPLIT, $0-32
	MOVQ   dst+0(FP), AX
	MOVQ   dst_len+8(FP), BX
	MOVSD  s+24(FP), X15
	UNPCKLPD X15, X15
	CMPQ   BX, $2
	JL     done_ms2
loop_ms2:
	MOVUPD (AX), X0
	MULPD  X15, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_ms2
done_ms2:
	RET

// reduceSumSqAVX2(a []float64) float64
// Computes sum of squares: sum(a[i]*a[i])
TEXT ·reduceSumSqAVX2(SB), NOSPLIT, $0-32
	MOVQ   a+0(FP), AX
	MOVQ   a_len+8(FP), BX
	VXORPD Y0, Y0, Y0
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
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
	MOVSD  X0, ret+24(FP)
	VZEROUPPER
	RET

// reduceSumSqSSE2(a []float64) float64
TEXT ·reduceSumSqSSE2(SB), NOSPLIT, $0-32
	MOVQ   a+0(FP), AX
	MOVQ   a_len+8(FP), BX
	XORPS  X0, X0
	CMPQ   BX, $2
	JL     done_ssq2
loop_ssq2:
	MOVUPD (AX), X1
	MULPD  X1, X1
	ADDPD  X1, X0
	ADDQ $16, AX
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_ssq2
done_ssq2:
	HADDPD X0, X0
	MOVSD  X0, ret+24(FP)
	RET

// signVecAVX2(dst, src []float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP),
//       src+24(FP), src_len+32(FP), src_cap+40(FP)
// Strategy: cmp > 0 → mask1, cmp < 0 → mask2, blend +1/-1/0 via masks.
TEXT ·signVecAVX2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src_len+32(FP), BX
	MOVQ src+24(FP), DI
	VXORPD Y15, Y15, Y15                // zero
	// broadcast +1.0 and -1.0
	MOVSD $0x3FF0000000000000, X13      // +1.0
	VBROADCASTSD X13, Y13
	MOVSD $0xBFF0000000000000, X14      // -1.0
	VBROADCASTSD X14, Y14
	CMPQ BX, $4
	JL   tail_sign
loop_sign:
	VMOVUPD  (DI), Y0
	VCMPPD   $1, Y15, Y0, Y1           // Y1 = (Y15 < Y0) i.e. src > 0
	VCMPPD   $1, Y0, Y15, Y2           // Y2 = (Y0 < Y15) i.e. src < 0
	VBLENDVPD Y1, Y13, Y15, Y3         // +1 where positive, else 0
	VBLENDVPD Y2, Y14, Y3,  Y3         // -1 where negative
	VMOVUPD  Y3, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_sign
tail_sign:
	VZEROUPPER
	RET

// signVecSSE2(dst, src []float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP),
//       src+24(FP), src_len+32(FP), src_cap+40(FP)
TEXT ·signVecSSE2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src_len+32(FP), BX
	MOVQ src+24(FP), DI
	XORPS X15, X15
	MOVSD $0x3FF0000000000000, X13     // +1.0
	SHUFPD $0, X13, X13
	MOVSD $0xBFF0000000000000, X14     // -1.0
	SHUFPD $0, X14, X14
	CMPQ BX, $2
	JL   tail_sign2
loop_sign2:
	MOVUPD (DI), X0
	MOVAPD X0, X1
	MOVAPD X0, X2
	CMPPD  $1, X15, X1                 // X1 = src > 0
	CMPPD  $1, X0,  X2                 // X2 = 0 > src (src < 0); note: args reversed
	// CMPPD $1 is LT_OS: dst[i] < src[i], so: X2: X15 < X0 means 0 < src → positive
	// We need negative: src < 0 → X15 > X0 → use pred $2 (LE) or swap
	// Redo: X1 = (X0 > X15): CMPPD $6 (NLE = GT)
	// X2 = (X15 > X0): compare X15 and X0 with NLE
	// Go asm doesn't have CMPPD with immediate easily — use CMPLTPD
	// Recompute cleanly:
	MOVAPD X15, X3
	CMPLTPD X0, X3                     // X3 = 0 < src (positive mask)
	MOVAPD  X0, X4
	CMPLTPD X15, X4                    // X4 = src < 0 (negative mask)
	ANDPD   X13, X3                    // +1 where positive
	ANDPD   X14, X4                    // -1 where negative
	ORPD    X4, X3
	MOVUPD  X3, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_sign2
tail_sign2:
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

// outerRowSSE2(dst, b []float64, scale float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP),
//       b+24(FP),  b_len+32(FP),  b_cap+40(FP),
//       scale+48(FP)
TEXT ·outerRowSSE2(SB), NOSPLIT, $0-56
	MOVQ  dst+0(FP), AX
	MOVQ  b_len+32(FP), BX
	MOVQ  b+24(FP), DI
	MOVSD scale+48(FP), X15
	SHUFPD $0, X15, X15
	CMPQ BX, $2
	JL   tail_or2
loop_or2:
	MOVUPD (DI), X0
	MULPD  X15, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_or2
tail_or2:
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

// addScaledVecSSE2(dst, src []float64, scale float64)
TEXT ·addScaledVecSSE2(SB), NOSPLIT, $0-56
	MOVQ  dst+0(FP), AX
	MOVQ  src_len+32(FP), BX
	MOVQ  src+24(FP), DI
	MOVSD scale+48(FP), X15
	SHUFPD $0, X15, X15
	CMPQ BX, $2
	JL   done_asv2
loop_asv2:
	MOVUPD (AX), X0
	MOVUPD (DI), X1
	MULPD  X15, X1
	ADDPD  X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_asv2
done_asv2:
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

// sqrtVecSSE2(dst, src []float64)
TEXT ·sqrtVecSSE2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src_len+32(FP), BX
	MOVQ src+24(FP), DI
	CMPQ BX, $2
	JL   done_sv3
loop_sv3:
	MOVUPD (DI), X0
	SQRTPD X0, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_sv3
done_sv3:
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

// addScalarVecSSE2(dst []float64, scalar float64)
TEXT ·addScalarVecSSE2(SB), NOSPLIT, $0-32
	MOVQ  dst+0(FP), AX
	MOVQ  dst_len+8(FP), BX
	MOVSD scalar+24(FP), X15
	SHUFPD $0, X15, X15
	CMPQ BX, $2
	JL   done_asa2
loop_asa2:
	MOVUPD (AX), X0
	ADDPD  X15, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_asa2
done_asa2:
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

// divVecSSE2(dst, a, b []float64)
TEXT ·divVecSSE2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ a_len+32(FP), BX
	MOVQ a+24(FP), DI
	MOVQ b+48(FP), SI
	CMPQ BX, $2
	JL   done_dv2
loop_dv2:
	MOVUPD (DI), X0
	MOVUPD (SI), X1
	DIVPD  X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_dv2
done_dv2:
	RET

// l2NormSqAVX2(a []float64) float64  — sum(a[i]^2)
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·l2NormSqAVX2(SB), NOSPLIT, $0-32
	MOVQ   a+0(FP), AX
	MOVQ   a_len+8(FP), BX
	VXORPD Y0, Y0, Y0
	CMPQ   BX, $4
	JL     done_l2a
loop_l2a:
	VMOVUPD (AX), Y1
	VFMADD231PD Y1, Y1, Y0
	ADDQ $32, AX
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_l2a
done_l2a:
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
	MOVSD  X0, ret+24(FP)
	VZEROUPPER
	RET

// l2NormSqSSE2(a []float64) float64
TEXT ·l2NormSqSSE2(SB), NOSPLIT, $0-32
	MOVQ  a+0(FP), AX
	MOVQ  a_len+8(FP), BX
	XORPS X0, X0
	CMPQ  BX, $2
	JL    done_l2b
loop_l2b:
	MOVUPD (AX), X1
	MULPD  X1, X1
	ADDPD  X1, X0
	ADDQ $16, AX
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_l2b
done_l2b:
	HADDPD X0, X0
	MOVSD  X0, ret+24(FP)
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

// clampVecSSE2(dst []float64, lo, hi float64)
TEXT ·clampVecSSE2(SB), NOSPLIT, $0-40
	MOVQ  dst+0(FP), AX
	MOVQ  dst_len+8(FP), BX
	MOVSD lo+24(FP), X14
	SHUFPD $0, X14, X14
	MOVSD hi+32(FP), X15
	SHUFPD $0, X15, X15
	CMPQ BX, $2
	JL   done_cv2
loop_cv2:
	MOVUPD (AX), X0
	MAXPD  X14, X0
	MINPD  X15, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_cv2
done_cv2:
	RET
