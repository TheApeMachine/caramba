#include "textflag.h"

// larsStepAVX2(out, velocity, params, grads []float64,
//              localLR, momentum, wd float64)
//   effGrad = grads + wd*params
//   v       = μ*v + localLR*effGrad
//   out     = params - v
TEXT ·larsStepAVX2(SB), NOSPLIT, $0-120
	MOVQ out+0(FP), AX
	MOVQ velocity+24(FP), R8
	MOVQ params+48(FP), R9
	MOVQ grads+72(FP), R10
	MOVQ out_len+8(FP), CX
	VBROADCASTSD localLR+96(FP), Y8
	VBROADCASTSD momentum+104(FP), Y9
	VBROADCASTSD wd+112(FP), Y10

	CMPQ CX, $4
	JL lars_avx2_tail
lars_avx2_loop:
	VMOVUPD (R8), Y0
	VMOVUPD (R9), Y1
	VMOVUPD (R10), Y2

	// effGrad = grads + wd*params
	VMOVAPD     Y2, Y3
	VFMADD231PD Y10, Y1, Y3

	// v = μ*v + localLR*effGrad
	VMULPD      Y9, Y0, Y0
	VFMADD231PD Y8, Y3, Y0
	VMOVUPD     Y0, (R8)

	// out = params - v
	VSUBPD  Y0, Y1, Y1
	VMOVUPD Y1, (AX)

	ADDQ $32, AX
	ADDQ $32, R8
	ADDQ $32, R9
	ADDQ $32, R10
	SUBQ $4, CX
	CMPQ CX, $4
	JGE lars_avx2_loop

lars_avx2_tail:
	CMPQ CX, $2
	JL lars_avx2_scalar
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVUPD (R10), X2
	MOVSD localLR+96(FP), X8
	SHUFPD $0, X8, X8
	MOVSD momentum+104(FP), X9
	SHUFPD $0, X9, X9
	MOVSD wd+112(FP), X10
	SHUFPD $0, X10, X10

	MOVAPD X2, X3
	MOVAPD X1, X11
	MULPD X10, X11
	ADDPD X11, X3

	MULPD X9, X0
	MOVAPD X3, X11
	MULPD X8, X11
	ADDPD X11, X0
	MOVUPD X0, (R8)

	SUBPD X0, X1
	MOVUPD X1, (AX)

	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $2, CX

lars_avx2_scalar:
	CMPQ CX, $0
	JLE lars_avx2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVSD (R10), X2
	MOVSD localLR+96(FP), X8
	MOVSD momentum+104(FP), X9
	MOVSD wd+112(FP), X10

	MOVAPD X2, X3
	MOVAPD X1, X11
	MULSD X10, X11
	ADDSD X11, X3

	MULSD X9, X0
	MOVAPD X3, X11
	MULSD X8, X11
	ADDSD X11, X0
	MOVSD X0, (R8)

	SUBSD X0, X1
	MOVSD X1, (AX)

lars_avx2_done:
	VZEROUPPER
	RET

// lambStepAVX2(out, m, v, params, grads []float64,
//              ratio, bc1Inv, bc2Inv, eps, wd float64)
//   m = β1*m + (1-β1)*g    (assumed already in m by caller)
//   v = β2*v + (1-β2)*g²   (assumed already in v by caller)
//   mHat = m * bc1Inv,  vHat = v * bc2Inv
//   update = mHat/(sqrt(vHat)+eps) + wd*params
//   out = params - ratio*update
TEXT ·lambStepAVX2(SB), NOSPLIT, $0-160
	MOVQ out+0(FP), AX
	MOVQ m+24(FP), R8
	MOVQ v+48(FP), R11
	MOVQ params+72(FP), R9
	MOVQ grads+96(FP), R10
	MOVQ out_len+8(FP), CX

	VBROADCASTSD ratio+120(FP), Y8
	VBROADCASTSD bc1Inv+128(FP), Y9
	VBROADCASTSD bc2Inv+136(FP), Y10
	VBROADCASTSD eps+144(FP), Y11
	VBROADCASTSD wd+152(FP), Y12

	CMPQ CX, $4
	JL lamb_avx2_tail
lamb_avx2_loop:
	VMOVUPD (R8), Y0                            // m
	VMOVUPD (R11), Y1                           // v
	VMOVUPD (R9), Y2                            // params

	VMULPD Y9, Y0, Y3                           // mHat
	VMULPD Y10, Y1, Y4                          // vHat

	VSQRTPD Y4, Y5
	VADDPD Y11, Y5, Y5                          // sqrt(vHat)+eps
	VDIVPD Y5, Y3, Y6                           // mHat / denom

	// update = (mHat/denom) + wd*params
	VFMADD231PD Y12, Y2, Y6                     // Y6 += wd*params

	VMOVAPD     Y2, Y7
	VFNMADD231PD Y8, Y6, Y7                     // Y7 = params - ratio*update
	VMOVUPD     Y7, (AX)

	ADDQ $32, AX
	ADDQ $32, R8
	ADDQ $32, R11
	ADDQ $32, R9
	ADDQ $32, R10
	SUBQ $4, CX
	CMPQ CX, $4
	JGE lamb_avx2_loop

lamb_avx2_tail:
	CMPQ CX, $2
	JL lamb_avx2_scalar
	MOVUPD (R8), X0
	MOVUPD (R11), X1
	MOVUPD (R9), X2
	MOVSD ratio+120(FP), X8
	SHUFPD $0, X8, X8
	MOVSD bc1Inv+128(FP), X9
	SHUFPD $0, X9, X9
	MOVSD bc2Inv+136(FP), X10
	SHUFPD $0, X10, X10
	MOVSD eps+144(FP), X11
	SHUFPD $0, X11, X11
	MOVSD wd+152(FP), X12
	SHUFPD $0, X12, X12

	MOVAPD X0, X3
	MULPD X9, X3
	MOVAPD X1, X4
	MULPD X10, X4

	MOVAPD X4, X5
	SQRTPD X5, X5
	ADDPD X11, X5
	MOVAPD X3, X6
	DIVPD X5, X6

	MOVAPD X2, X13
	MULPD X12, X13
	ADDPD X13, X6

	MOVAPD X2, X7
	MOVAPD X6, X13
	MULPD X8, X13
	SUBPD X13, X7
	MOVUPD X7, (AX)

	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R11
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $2, CX

lamb_avx2_scalar:
	CMPQ CX, $0
	JLE lamb_avx2_done
	MOVSD (R8), X0
	MOVSD (R11), X1
	MOVSD (R9), X2
	MOVSD ratio+120(FP), X8
	MOVSD bc1Inv+128(FP), X9
	MOVSD bc2Inv+136(FP), X10
	MOVSD eps+144(FP), X11
	MOVSD wd+152(FP), X12

	MOVAPD X0, X3
	MULSD X9, X3
	MOVAPD X1, X4
	MULSD X10, X4

	MOVAPD X4, X5
	SQRTSD X5, X5
	ADDSD X11, X5
	MOVAPD X3, X6
	DIVSD X5, X6

	MOVAPD X2, X13
	MULSD X12, X13
	ADDSD X13, X6

	MOVAPD X2, X7
	MOVAPD X6, X13
	MULSD X8, X13
	SUBSD X13, X7
	MOVSD X7, (AX)

lamb_avx2_done:
	VZEROUPPER
	RET

// lambEMAAVX2(m, v, grads []float64, beta1, oneMinusBeta1, beta2, oneMinusBeta2 float64)
//   m = β1*m + (1-β1)*g
//   v = β2*v + (1-β2)*g²
TEXT ·lambEMAAVX2(SB), NOSPLIT, $0-104
	MOVQ m+0(FP), R8
	MOVQ v+24(FP), R11
	MOVQ grads+48(FP), R10
	MOVQ m_len+8(FP), CX

	VBROADCASTSD beta1+72(FP), Y8
	VBROADCASTSD oneMinusBeta1+80(FP), Y9
	VBROADCASTSD beta2+88(FP), Y10
	VBROADCASTSD oneMinusBeta2+96(FP), Y11

	CMPQ CX, $4
	JL lema_avx2_tail
lema_avx2_loop:
	VMOVUPD (R8), Y0
	VMOVUPD (R11), Y1
	VMOVUPD (R10), Y2

	VMULPD Y2, Y2, Y3
	VMULPD Y8, Y0, Y0
	VFMADD231PD Y9, Y2, Y0
	VMOVUPD Y0, (R8)
	VMULPD Y10, Y1, Y1
	VFMADD231PD Y11, Y3, Y1
	VMOVUPD Y1, (R11)

	ADDQ $32, R8
	ADDQ $32, R11
	ADDQ $32, R10
	SUBQ $4, CX
	CMPQ CX, $4
	JGE lema_avx2_loop

lema_avx2_tail:
	CMPQ CX, $2
	JL lema_avx2_scalar
	MOVUPD (R8), X0
	MOVUPD (R11), X1
	MOVUPD (R10), X2
	MOVSD beta1+72(FP), X8
	SHUFPD $0, X8, X8
	MOVSD oneMinusBeta1+80(FP), X9
	SHUFPD $0, X9, X9
	MOVSD beta2+88(FP), X10
	SHUFPD $0, X10, X10
	MOVSD oneMinusBeta2+96(FP), X11
	SHUFPD $0, X11, X11

	MOVAPD X2, X3
	MULPD X3, X3
	MULPD X8, X0
	MOVAPD X2, X4
	MULPD X9, X4
	ADDPD X4, X0
	MOVUPD X0, (R8)
	MULPD X10, X1
	MOVAPD X3, X4
	MULPD X11, X4
	ADDPD X4, X1
	MOVUPD X1, (R11)

	ADDQ $16, R8
	ADDQ $16, R11
	ADDQ $16, R10
	SUBQ $2, CX

lema_avx2_scalar:
	CMPQ CX, $0
	JLE lema_avx2_done
	MOVSD (R8), X0
	MOVSD (R11), X1
	MOVSD (R10), X2
	MOVSD beta1+72(FP), X8
	MOVSD oneMinusBeta1+80(FP), X9
	MOVSD beta2+88(FP), X10
	MOVSD oneMinusBeta2+96(FP), X11

	MOVAPD X2, X3
	MULSD X3, X3
	MULSD X8, X0
	MOVAPD X2, X4
	MULSD X9, X4
	ADDSD X4, X0
	MOVSD X0, (R8)
	MULSD X10, X1
	MOVAPD X3, X4
	MULSD X11, X4
	ADDSD X4, X1
	MOVSD X1, (R11)

lema_avx2_done:
	VZEROUPPER
	RET

// lambL2NormSqAVX2(a []float64) float64
TEXT ·lambL2NormSqAVX2(SB), NOSPLIT, $0-32
	MOVQ a+0(FP), AX
	MOVQ a_len+8(FP), CX
	VXORPD Y0, Y0, Y0
	CMPQ CX, $4
	JL ll2_avx2_tail
ll2_avx2_loop:
	VMOVUPD (AX), Y1
	VFMADD231PD Y1, Y1, Y0
	ADDQ $32, AX
	SUBQ $4, CX
	CMPQ CX, $4
	JGE ll2_avx2_loop
	VEXTRACTF128 $1, Y0, X2
	VADDPD X2, X0, X0
	VHADDPD X0, X0, X0
ll2_avx2_tail:
	CMPQ CX, $2
	JL ll2_avx2_scalar
	MOVUPD (AX), X1
	MULPD X1, X1
	HADDPD X1, X1
	ADDSD X1, X0
	ADDQ $16, AX
	SUBQ $2, CX
ll2_avx2_scalar:
	CMPQ CX, $0
	JLE ll2_avx2_done
	MOVSD (AX), X1
	MULSD X1, X1
	ADDSD X1, X0
ll2_avx2_done:
	MOVSD X0, ret+24(FP)
	VZEROUPPER
	RET

// lambUpdateNormSqAVX2(m, v, params []float64, bc1Inv, bc2Inv, eps, wd float64) float64
// Returns sum of (mHat/(sqrt(vHat)+eps) + wd*params)² without writing anywhere.
TEXT ·lambUpdateNormSqAVX2(SB), NOSPLIT, $0-112
	MOVQ m+0(FP), R8
	MOVQ v+24(FP), R11
	MOVQ params+48(FP), R9
	MOVQ m_len+8(FP), CX

	VBROADCASTSD bc1Inv+72(FP), Y9
	VBROADCASTSD bc2Inv+80(FP), Y10
	VBROADCASTSD eps+88(FP), Y11
	VBROADCASTSD wd+96(FP), Y12
	VXORPD Y15, Y15, Y15

	CMPQ CX, $4
	JL luns_avx2_tail
luns_avx2_loop:
	VMOVUPD (R8), Y0
	VMOVUPD (R11), Y1
	VMOVUPD (R9), Y2

	VMULPD Y9, Y0, Y3
	VMULPD Y10, Y1, Y4
	VSQRTPD Y4, Y5
	VADDPD Y11, Y5, Y5
	VDIVPD Y5, Y3, Y6
	VFMADD231PD Y12, Y2, Y6
	VFMADD231PD Y6, Y6, Y15

	ADDQ $32, R8
	ADDQ $32, R11
	ADDQ $32, R9
	SUBQ $4, CX
	CMPQ CX, $4
	JGE luns_avx2_loop
	VEXTRACTF128 $1, Y15, X14
	VADDPD X14, X15, X15
	VHADDPD X15, X15, X15

luns_avx2_tail:
	CMPQ CX, $2
	JL luns_avx2_scalar
	MOVUPD (R8), X0
	MOVUPD (R11), X1
	MOVUPD (R9), X2
	MOVSD bc1Inv+72(FP), X9
	SHUFPD $0, X9, X9
	MOVSD bc2Inv+80(FP), X10
	SHUFPD $0, X10, X10
	MOVSD eps+88(FP), X11
	SHUFPD $0, X11, X11
	MOVSD wd+96(FP), X12
	SHUFPD $0, X12, X12

	MOVAPD X0, X3
	MULPD X9, X3
	MOVAPD X1, X4
	MULPD X10, X4
	MOVAPD X4, X5
	SQRTPD X5, X5
	ADDPD X11, X5
	MOVAPD X3, X6
	DIVPD X5, X6
	MOVAPD X2, X13
	MULPD X12, X13
	ADDPD X13, X6
	MOVAPD X6, X14
	MULPD X14, X14
	HADDPD X14, X14
	ADDSD X14, X15
	ADDQ $16, R8
	ADDQ $16, R11
	ADDQ $16, R9
	SUBQ $2, CX

luns_avx2_scalar:
	CMPQ CX, $0
	JLE luns_avx2_done
	MOVSD (R8), X0
	MOVSD (R11), X1
	MOVSD (R9), X2
	MOVSD bc1Inv+72(FP), X9
	MOVSD bc2Inv+80(FP), X10
	MOVSD eps+88(FP), X11
	MOVSD wd+96(FP), X12

	MOVAPD X0, X3
	MULSD X9, X3
	MOVAPD X1, X4
	MULSD X10, X4
	MOVAPD X4, X5
	SQRTSD X5, X5
	ADDSD X11, X5
	MOVAPD X3, X6
	DIVSD X5, X6
	MOVAPD X2, X13
	MULSD X12, X13
	ADDSD X13, X6
	MOVAPD X6, X14
	MULSD X14, X14
	ADDSD X14, X15

luns_avx2_done:
	MOVSD X15, ret+104(FP)
	VZEROUPPER
	RET
