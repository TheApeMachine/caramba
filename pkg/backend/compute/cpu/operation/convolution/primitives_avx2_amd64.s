#include "textflag.h"

// dotProductAVX2(a, b []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP),
//       b+24(FP), b_len+32(FP), b_cap+40(FP),
//       ret+48(FP)
TEXT ·dotProductAVX2(SB), NOSPLIT, $0-56
	MOVQ  a+0(FP), AX
	MOVQ  a_len+8(FP), BX
	MOVQ  b+24(FP), DI
	VXORPD Y0, Y0, Y0
	CMPQ   BX, $4
	JL     scalar_dp
loop_dp:
	VMOVUPD (AX), Y1
	VMOVUPD (DI), Y2
	VFMADD231PD Y1, Y2, Y0
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_dp
scalar_dp:
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
	MOVSD X0, ret+48(FP)
	VZEROUPPER
	RET

// dotProductSSE2(a, b []float64) float64
TEXT ·dotProductSSE2(SB), NOSPLIT, $0-56
	MOVQ  a+0(FP), AX
	MOVQ  a_len+8(FP), BX
	MOVQ  b+24(FP), DI
	XORPS X0, X0
	CMPQ  BX, $2
	JL    scalar_dp2
loop_dp2:
	MOVUPD (AX), X1
	MOVUPD (DI), X2
	MULPD  X2, X1
	ADDPD  X1, X0
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_dp2
scalar_dp2:
	HADDPD X0, X0
	MOVSD  X0, ret+48(FP)
	RET

// scaledAddAVX2(dst, src []float64, scale float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP),
//       src+24(FP), src_len+32(FP), src_cap+40(FP),
//       scale+48(FP)
TEXT ·scaledAddAVX2(SB), NOSPLIT, $0-56
	MOVQ     dst+0(FP), AX
	MOVQ     src_len+32(FP), BX
	MOVQ     src+24(FP), DI
	VMOVSD   scale+48(FP), X15
	VBROADCASTSD X15, Y15
	CMPQ     BX, $4
	JL       done_sa
loop_sa:
	VMOVUPD (DI), Y1
	VMOVUPD (AX), Y0
	VFMADD231PD Y1, Y15, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_sa
done_sa:
	VZEROUPPER
	RET

// scaledAddSSE2(dst, src []float64, scale float64)
TEXT ·scaledAddSSE2(SB), NOSPLIT, $0-56
	MOVQ   dst+0(FP), AX
	MOVQ   src_len+32(FP), BX
	MOVQ   src+24(FP), DI
	MOVSD  scale+48(FP), X15
	UNPCKLPD X15, X15
	CMPQ   BX, $2
	JL     done_sa2
loop_sa2:
	MOVUPD (DI), X1
	MOVUPD (AX), X0
	MULPD  X15, X1
	ADDPD  X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_sa2
done_sa2:
	RET
