#include "textflag.h"

// LeakyReLUAVX2(dst, src []float64, alpha float64)
// ABI0: dst+0(FP)=ptr, dst_len+8(FP)=len, dst_cap+16(FP)=cap,
//       src_base+24(FP)=ptr, src_len+32(FP)=len, src_cap+40(FP)=cap,
//       alpha+48(FP)=float64
//
// out[i] = max(alpha*x, x)   (correct for alpha in (0,1))

TEXT ·LeakyReLUAVX2(SB), NOSPLIT, $0-56
	MOVQ  dst+0(FP), AX
	MOVQ  src_len+32(FP), SI
	MOVQ  src_base+24(FP), DI
	CMPQ  SI, $0
	JLE   done

	// Load alpha scalar, broadcast to Y14
	MOVSD alpha+48(FP), X0
	VBROADCASTSD X0, Y14

loop:
	VMOVUPD (DI), Y1
	VMULPD  Y14, Y1, Y2        // alpha*x
	VMAXPD  Y2, Y1, Y3         // max(alpha*x, x)
	VMOVUPD Y3, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, SI
	CMPQ SI, $4
	JGE  loop
done:
	VZEROUPPER
	RET
