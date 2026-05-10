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
