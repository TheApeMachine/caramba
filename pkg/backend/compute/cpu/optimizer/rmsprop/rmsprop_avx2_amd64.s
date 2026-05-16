#include "textflag.h"

// rmspropPlainAVX2(out, v, params, grads []float64,
//                  lr, alpha, oneMinusAlpha, eps, wd float64)
//   geff = g + wd*p
//   v    = α*v + (1-α)*geff²
//   denom = sqrt(v) + eps
//   out = p - lr * geff / denom
TEXT ·rmspropPlainAVX2(SB), NOSPLIT, $0-136
	MOVQ out+0(FP), AX
	MOVQ v+24(FP), R8
	MOVQ params+48(FP), R9
	MOVQ grads+72(FP), R10
	MOVQ out_len+8(FP), CX

	VBROADCASTSD lr+96(FP), Y8
	VBROADCASTSD alpha+104(FP), Y9
	VBROADCASTSD oneMinusAlpha+112(FP), Y10
	VBROADCASTSD eps+120(FP), Y11
	VBROADCASTSD wd+128(FP), Y12

	CMPQ CX, $4
	JL   rmsp_avx2_tail
rmsp_avx2_loop:
	VMOVUPD (R8), Y0                            // v
	VMOVUPD (R9), Y1                            // params
	VMOVUPD (R10), Y2                           // grads

	// geff = g + wd*p
	VMOVAPD     Y2, Y3
	VFMADD231PD Y12, Y1, Y3

	// v = α*v + (1-α)*geff²
	VMULPD      Y3, Y3, Y4
	VMULPD      Y9, Y0, Y0
	VFMADD231PD Y10, Y4, Y0
	VMOVUPD     Y0, (R8)

	// denom = sqrt(v) + eps
	VSQRTPD Y0, Y5
	VADDPD  Y11, Y5, Y5
	// upd = geff / denom
	VDIVPD Y5, Y3, Y6
	// out = p - lr*upd
	VFNMADD231PD Y8, Y6, Y1
	VMOVUPD      Y1, (AX)

	ADDQ $32, AX
	ADDQ $32, R8
	ADDQ $32, R9
	ADDQ $32, R10
	SUBQ $4, CX
	CMPQ CX, $4
	JGE  rmsp_avx2_loop

rmsp_avx2_tail:
	CMPQ CX, $2
	JL   rmsp_avx2_scalar
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVUPD (R10), X2

	MOVSD lr+96(FP), X8
	SHUFPD $0, X8, X8
	MOVSD alpha+104(FP), X9
	SHUFPD $0, X9, X9
	MOVSD oneMinusAlpha+112(FP), X10
	SHUFPD $0, X10, X10
	MOVSD eps+120(FP), X11
	SHUFPD $0, X11, X11
	MOVSD wd+128(FP), X12
	SHUFPD $0, X12, X12

	MOVAPD X2, X3
	MOVAPD X1, X13
	MULPD X12, X13
	ADDPD X13, X3

	MOVAPD X3, X4
	MULPD X4, X4
	MULPD X9, X0
	MOVAPD X4, X13
	MULPD X10, X13
	ADDPD X13, X0
	MOVUPD X0, (R8)

	MOVAPD X0, X5
	SQRTPD X5, X5
	ADDPD X11, X5
	MOVAPD X3, X6
	DIVPD X5, X6
	MULPD X8, X6
	SUBPD X6, X1
	MOVUPD X1, (AX)

	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $2, CX

rmsp_avx2_scalar:
	CMPQ CX, $0
	JLE  rmsp_avx2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVSD (R10), X2

	MOVSD lr+96(FP), X8
	MOVSD alpha+104(FP), X9
	MOVSD oneMinusAlpha+112(FP), X10
	MOVSD eps+120(FP), X11
	MOVSD wd+128(FP), X12

	MOVAPD X2, X3
	MOVAPD X1, X13
	MULSD X12, X13
	ADDSD X13, X3

	MOVAPD X3, X4
	MULSD X4, X4
	MULSD X9, X0
	MOVAPD X4, X13
	MULSD X10, X13
	ADDSD X13, X0
	MOVSD X0, (R8)

	MOVAPD X0, X5
	SQRTSD X5, X5
	ADDSD X11, X5
	MOVAPD X3, X6
	DIVSD X5, X6
	MULSD X8, X6
	SUBSD X6, X1
	MOVSD X1, (AX)

rmsp_avx2_done:
	VZEROUPPER
	RET
