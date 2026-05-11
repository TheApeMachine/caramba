#include "textflag.h"

// bindAVX2(dst, a, b []float64)
// ABI0: dst+0(FP)..16, a+24(FP)..40, b+48(FP)..64
TEXT ·bindAVX2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ a_len+32(FP), BX
	MOVQ a+24(FP), DI
	MOVQ b+48(FP), SI
	CMPQ BX, $4
	JL   done_ba
loop_ba:
	VMOVUPD (DI), Y0
	VMOVUPD (SI), Y1
	VMULPD  Y1, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_ba
done_ba:
	VZEROUPPER
	RET

// dotReduceAVX2(a, b []float64) float64
// ABI0: a+0(FP)..16, b+24(FP)..40, ret+48(FP)
TEXT ·dotReduceAVX2(SB), NOSPLIT, $0-56
	MOVQ   a+0(FP), AX
	MOVQ   a_len+8(FP), BX
	MOVQ   b+24(FP), DI
	VXORPD Y0, Y0, Y0
	CMPQ   BX, $4
	JL     done_da
loop_da:
	VMOVUPD (AX), Y1
	VMOVUPD (DI), Y2
	VMULPD  Y2, Y1, Y1
	VADDPD  Y1, Y0, Y0
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_da
done_da:
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
	MOVSD  X0, ret+48(FP)
	VZEROUPPER
	RET

// addInPlaceAVX2(dst, src []float64)
// ABI0: dst+0(FP)..16, src+24(FP)..40
TEXT ·addInPlaceAVX2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src_len+32(FP), BX
	MOVQ src+24(FP), DI
	CMPQ BX, $4
	JL   done_aip
loop_aip:
	VMOVUPD (AX), Y0
	VMOVUPD (DI), Y1
	VADDPD  Y1, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_aip
done_aip:
	VZEROUPPER
	RET

// mulScalarVecAVX2(dst []float64, s float64)
// ABI0: dst+0(FP), dst_len+8(FP), dst_cap+16(FP), s+24(FP)
TEXT ·mulScalarVecAVX2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), AX
	MOVQ dst_len+8(FP), BX
	VBROADCASTSD s+24(FP), Y15
	CMPQ BX, $4
	JL   done_msv
loop_msv:
	VMOVUPD (AX), Y0
	VMULPD  Y15, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_msv
done_msv:
	VZEROUPPER
	RET

// reduceSumSqAVX2(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceSumSqAVX2(SB), NOSPLIT, $0-32
	MOVQ   a+0(FP), AX
	MOVQ   a_len+8(FP), BX
	VXORPD Y0, Y0, Y0
	CMPQ   BX, $4
	JL     done_rss
loop_rss:
	VMOVUPD (AX), Y1
	VMULPD  Y1, Y1, Y1
	VADDPD  Y1, Y0, Y0
	ADDQ $32, AX
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_rss
done_rss:
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
	MOVSD  X0, ret+24(FP)
	VZEROUPPER
	RET
