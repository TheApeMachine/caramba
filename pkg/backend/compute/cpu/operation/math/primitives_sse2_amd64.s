#include "textflag.h"

DATA ·signOne_amd64+0(SB)/8, $1.0
GLOBL ·signOne_amd64(SB), RODATA, $8
DATA ·signNegOne_amd64+0(SB)/8, $-1.0
GLOBL ·signNegOne_amd64(SB), RODATA, $8

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

// signVecSSE2(dst, src []float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP),
//       src+24(FP), src_len+32(FP), src_cap+40(FP)
TEXT ·signVecSSE2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), DI
	MOVQ src_len+32(FP), BX
	XORPD X1, X1
	MOVSD ·signOne_amd64(SB), X2
	SHUFPD $0, X2, X2
	MOVSD ·signNegOne_amd64(SB), X3
	SHUFPD $0, X3, X3
	CMPQ BX, $2
	JL   done_sv_sse2
loop_sv_sse2:
	MOVUPD (DI), X0
	MOVAPD X1, X4
	CMPPD X0, X4, $1
	MOVAPD X0, X5
	CMPPD X1, X5, $1
	ANDPD  X2, X4
	ANDPD  X3, X5
	ORPD   X5, X4
	MOVUPD X4, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_sv_sse2
done_sv_sse2:
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
