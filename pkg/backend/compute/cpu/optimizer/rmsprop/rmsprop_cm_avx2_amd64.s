#include "textflag.h"

// rmspropCenteredMomentumAVX2(out, v, gAvg, buf, params, grads []float64,
//                             lr, alpha, oneMinusAlpha, eps, momentum, wd float64)
//   geff  = g + wd*p
//   v     = α*v + (1-α)*geff²
//   gAvg  = α*gAvg + (1-α)*geff
//   denom = sqrt(v - gAvg²) + eps
//   buf   = μ*buf + geff/denom
//   out   = p - lr*buf
TEXT ·rmspropCenteredMomentumAVX2(SB), NOSPLIT, $0-192
	MOVQ out+0(FP), AX
	MOVQ v+24(FP), R8
	MOVQ gAvg+48(FP), R11
	MOVQ buf+72(FP), R12
	MOVQ params+96(FP), R9
	MOVQ grads+120(FP), R10
	MOVQ out_len+8(FP), CX

	VBROADCASTSD lr+144(FP), Y8
	VBROADCASTSD alpha+152(FP), Y9
	VBROADCASTSD oneMinusAlpha+160(FP), Y10
	VBROADCASTSD eps+168(FP), Y11
	VBROADCASTSD momentum+176(FP), Y12
	VBROADCASTSD wd+184(FP), Y13

	CMPQ CX, $4
	JL   rmscm_avx2_tail
rmscm_avx2_loop:
	VMOVUPD (R8), Y0                            // v
	VMOVUPD (R11), Y1                           // gAvg
	VMOVUPD (R12), Y14                          // buf
	VMOVUPD (R9), Y2                            // params
	VMOVUPD (R10), Y3                           // grads

	VMOVAPD     Y3, Y4
	VFMADD231PD Y13, Y2, Y4                     // geff

	VMULPD      Y4, Y4, Y5
	VMULPD      Y9, Y0, Y0
	VFMADD231PD Y10, Y5, Y0
	VMOVUPD     Y0, (R8)

	VMULPD      Y9, Y1, Y1
	VFMADD231PD Y10, Y4, Y1
	VMOVUPD     Y1, (R11)

	VMULPD  Y1, Y1, Y6
	VSUBPD  Y6, Y0, Y6
	VSQRTPD Y6, Y6
	VADDPD  Y11, Y6, Y6
	VDIVPD  Y6, Y4, Y7

	VMULPD      Y12, Y14, Y14
	VADDPD      Y7, Y14, Y14
	VMOVUPD     Y14, (R12)

	VMULPD  Y8, Y14, Y7
	VSUBPD  Y7, Y2, Y2
	VMOVUPD Y2, (AX)

	ADDQ $32, AX
	ADDQ $32, R8
	ADDQ $32, R11
	ADDQ $32, R12
	ADDQ $32, R9
	ADDQ $32, R10
	SUBQ $4, CX
	CMPQ CX, $4
	JGE  rmscm_avx2_loop

rmscm_avx2_tail:
	CMPQ CX, $2
	JL   rmscm_avx2_scalar
	MOVUPD (R8), X0
	MOVUPD (R11), X1
	MOVUPD (R12), X14
	MOVUPD (R9), X2
	MOVUPD (R10), X3

	MOVSD lr+144(FP), X8
	SHUFPD $0, X8, X8
	MOVSD alpha+152(FP), X9
	SHUFPD $0, X9, X9
	MOVSD oneMinusAlpha+160(FP), X10
	SHUFPD $0, X10, X10
	MOVSD eps+168(FP), X11
	SHUFPD $0, X11, X11
	MOVSD momentum+176(FP), X12
	SHUFPD $0, X12, X12
	MOVSD wd+184(FP), X13
	SHUFPD $0, X13, X13

	MOVAPD X3, X4
	MOVAPD X2, X15
	MULPD X13, X15
	ADDPD X15, X4

	MOVAPD X4, X5
	MULPD X5, X5
	MULPD X9, X0
	MOVAPD X5, X15
	MULPD X10, X15
	ADDPD X15, X0
	MOVUPD X0, (R8)

	MULPD X9, X1
	MOVAPD X4, X15
	MULPD X10, X15
	ADDPD X15, X1
	MOVUPD X1, (R11)

	MOVAPD X1, X6
	MULPD X6, X6
	MOVAPD X0, X7
	SUBPD X6, X7
	SQRTPD X7, X7
	ADDPD X11, X7
	MOVAPD X4, X15
	DIVPD X7, X15

	MULPD X12, X14
	ADDPD X15, X14
	MOVUPD X14, (R12)

	MOVAPD X14, X7
	MULPD X8, X7
	SUBPD X7, X2
	MOVUPD X2, (AX)

	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R11
	ADDQ $16, R12
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $2, CX

rmscm_avx2_scalar:
	CMPQ CX, $0
	JLE  rmscm_avx2_done
	MOVSD (R8), X0
	MOVSD (R11), X1
	MOVSD (R12), X14
	MOVSD (R9), X2
	MOVSD (R10), X3

	MOVSD lr+144(FP), X8
	MOVSD alpha+152(FP), X9
	MOVSD oneMinusAlpha+160(FP), X10
	MOVSD eps+168(FP), X11
	MOVSD momentum+176(FP), X12
	MOVSD wd+184(FP), X13

	MOVAPD X3, X4
	MOVAPD X2, X15
	MULSD X13, X15
	ADDSD X15, X4

	MOVAPD X4, X5
	MULSD X5, X5
	MULSD X9, X0
	MOVAPD X5, X15
	MULSD X10, X15
	ADDSD X15, X0
	MOVSD X0, (R8)

	MULSD X9, X1
	MOVAPD X4, X15
	MULSD X10, X15
	ADDSD X15, X1
	MOVSD X1, (R11)

	MOVAPD X1, X6
	MULSD X6, X6
	MOVAPD X0, X7
	SUBSD X6, X7
	SQRTSD X7, X7
	ADDSD X11, X7
	MOVAPD X4, X15
	DIVSD X7, X15

	MULSD X12, X14
	ADDSD X15, X14
	MOVSD X14, (R12)

	MOVAPD X14, X7
	MULSD X8, X7
	SUBSD X7, X2
	MOVSD X2, (AX)

rmscm_avx2_done:
	VZEROUPPER
	RET
