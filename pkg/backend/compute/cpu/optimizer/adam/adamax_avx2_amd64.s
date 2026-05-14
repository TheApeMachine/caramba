#include "textflag.h"

DATA ·adamaxAbsMask+0(SB)/8, $0x7FFFFFFFFFFFFFFF
GLOBL ·adamaxAbsMask(SB), RODATA, $8

// adamaxStepAVX2(out, m, u, params, grads []float64,
//                beta1, oneMinusBeta1, beta2, lrT, eps float64)
//
//   m[i] = β1*m[i] + (1-β1)*g[i]
//   u[i] = max(β2*u[i], |g[i]|)
//   out[i] = params[i] - lrT * m[i] / (u[i] + ε)
TEXT ·adamaxStepAVX2(SB), NOSPLIT, $0-160
	MOVQ out+0(FP), AX
	MOVQ m+24(FP), R8
	MOVQ u+48(FP), R9
	MOVQ params+72(FP), R10
	MOVQ grads+96(FP), R11
	MOVQ out_len+8(FP), CX

	VBROADCASTSD beta1+120(FP), Y8
	VBROADCASTSD oneMinusBeta1+128(FP), Y9
	VBROADCASTSD beta2+136(FP), Y10
	VBROADCASTSD lrT+144(FP), Y11
	VBROADCASTSD eps+152(FP), Y12
	VBROADCASTSD ·adamaxAbsMask(SB), Y13

	CMPQ CX, $4
	JL   adamax_avx2_tail
adamax_avx2_loop:
	VMOVUPD (R8), Y0                      // m
	VMOVUPD (R9), Y1                      // u
	VMOVUPD (R10), Y2                     // params
	VMOVUPD (R11), Y3                     // g

	// m = β1*m + (1-β1)*g
	VMULPD       Y8, Y0, Y0
	VFMADD231PD  Y9, Y3, Y0
	VMOVUPD      Y0, (R8)

	// u = max(β2*u, |g|)
	VMULPD  Y10, Y1, Y1
	VANDPD  Y13, Y3, Y4                  // |g|
	VMAXPD  Y4, Y1, Y1
	VMOVUPD Y1, (R9)

	// denom = u + eps
	VADDPD Y12, Y1, Y5
	// upd = m / denom
	VDIVPD Y5, Y0, Y6
	// out = params - lrT*upd
	VMOVAPD      Y2, Y7
	VFNMADD231PD Y11, Y6, Y7
	VMOVUPD      Y7, (AX)

	ADDQ $32, AX
	ADDQ $32, R8
	ADDQ $32, R9
	ADDQ $32, R10
	ADDQ $32, R11
	SUBQ $4, CX
	CMPQ CX, $4
	JGE  adamax_avx2_loop

adamax_avx2_tail:
	CMPQ CX, $2
	JL   adamax_avx2_scalar

	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVUPD (R10), X2
	MOVUPD (R11), X3

	MOVSD beta1+120(FP), X8
	SHUFPD $0, X8, X8
	MOVSD oneMinusBeta1+128(FP), X9
	SHUFPD $0, X9, X9
	MOVSD beta2+136(FP), X10
	SHUFPD $0, X10, X10
	MOVSD lrT+144(FP), X11
	SHUFPD $0, X11, X11
	MOVSD eps+152(FP), X12
	SHUFPD $0, X12, X12
	MOVSD ·adamaxAbsMask(SB), X13
	SHUFPD $0, X13, X13

	MULPD X8, X0
	MOVAPD X3, X5
	MULPD X9, X5
	ADDPD X5, X0
	MOVUPD X0, (R8)

	MULPD X10, X1
	MOVAPD X3, X4
	ANDPD X13, X4
	MAXPD X4, X1
	MOVUPD X1, (R9)

	MOVAPD X1, X5
	ADDPD X12, X5
	MOVAPD X0, X6
	DIVPD X5, X6
	MULPD X11, X6
	SUBPD X6, X2
	MOVUPD X2, (AX)

	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	ADDQ $16, R11
	SUBQ $2, CX

adamax_avx2_scalar:
	CMPQ CX, $0
	JLE  adamax_avx2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVSD (R10), X2
	MOVSD (R11), X3

	MOVSD beta1+120(FP), X8
	MOVSD oneMinusBeta1+128(FP), X9
	MOVSD beta2+136(FP), X10
	MOVSD lrT+144(FP), X11
	MOVSD eps+152(FP), X12
	MOVSD ·adamaxAbsMask(SB), X13

	MULSD X8, X0
	MOVAPD X3, X5
	MULSD X9, X5
	ADDSD X5, X0
	MOVSD X0, (R8)

	MULSD X10, X1
	MOVAPD X3, X4
	ANDPD X13, X4
	MAXSD X4, X1
	MOVSD X1, (R9)

	MOVAPD X1, X5
	ADDSD X12, X5
	MOVAPD X0, X6
	DIVSD X5, X6
	MULSD X11, X6
	SUBSD X6, X2
	MOVSD X2, (AX)

adamax_avx2_done:
	VZEROUPPER
	RET
