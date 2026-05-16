#include "textflag.h"

// dotProductAVX2(a, b []float64) float64
// ABI0: a+0(FP)=ptr, a_len+8(FP)=len, a_cap+16(FP)=cap,
//       b+24(FP)=ptr, b_len+32(FP)=len, b_cap+40(FP)=cap,
//       ret+48(FP)=float64
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
	CMPQ BX, $0
	JLE  done_dp
tail_dp:
	VMOVSD (AX), X1
	VMOVSD (DI), X2
	VMULSD X2, X1, X1
	VADDSD X1, X0, X0
	ADDQ $8, AX
	ADDQ $8, DI
	DECQ BX
	JNZ  tail_dp
done_dp:
	MOVSD X0, ret+48(FP)
	VZEROUPPER
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
	JL       tail_sa
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
tail_sa:
	CMPQ BX, $0
	JLE  done_sa
tail_sa_loop:
	VMOVSD (DI), X1
	VMULSD X15, X1, X1
	VMOVSD (AX), X0
	VADDSD X1, X0, X0
	VMOVSD X0, (AX)
	ADDQ $8, AX
	ADDQ $8, DI
	DECQ BX
	JNZ  tail_sa_loop
done_sa:
	VZEROUPPER
	RET

// reduceMaxAVX2(a []float64) float64
// ABI0: a+0(FP), a_len+8(FP), a_cap+16(FP), ret+24(FP)
TEXT ·reduceMaxAVX2(SB), NOSPLIT, $0-32
	MOVQ    a+0(FP), AX
	MOVQ    a_len+8(FP), BX
	CMPQ    BX, $4
	JL      scalar_rm_init
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
	JMP    tail_rm
scalar_rm_init:
	VMOVSD (AX), X0
	ADDQ   $8, AX
	DECQ   BX
tail_rm:
	CMPQ BX, $0
	JLE  done_rm
tail_rm_loop:
	VMOVSD (AX), X1
	VMAXSD X1, X0, X0
	ADDQ $8, AX
	DECQ BX
	JNZ  tail_rm_loop
done_rm:
	MOVSD  X0, ret+24(FP)
	VZEROUPPER
	RET

// reduceSumAVX2(a []float64) float64
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
	CMPQ BX, $0
	JLE  done_rs_ret
tail_rs:
	VMOVSD (AX), X1
	VADDSD X1, X0, X0
	ADDQ $8, AX
	DECQ BX
	JNZ  tail_rs
done_rs_ret:
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
	JL       tail_ds
loop_ds:
	VMOVUPD (AX), Y0
	VDIVPD  Y15, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  loop_ds
tail_ds:
	CMPQ BX, $0
	JLE  done_ds
tail_ds_loop:
	VMOVSD (AX), X0
	VDIVSD X15, X0, X0
	VMOVSD X0, (AX)
	ADDQ $8, AX
	DECQ BX
	JNZ  tail_ds_loop
done_ds:
	VZEROUPPER
	RET
