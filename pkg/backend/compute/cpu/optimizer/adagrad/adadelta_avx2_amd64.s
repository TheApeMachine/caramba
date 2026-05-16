#include "textflag.h"

// adadeltaStepAVX2(out, eg2, edp2, params, grads []float64,
//                  rho, oneMinusRho, eps, wd float64)
TEXT ·adadeltaStepAVX2(SB), NOSPLIT, $0-152
	MOVQ out+0(FP), AX
	MOVQ eg2+24(FP), R8
	MOVQ edp2+48(FP), R9
	MOVQ params+72(FP), R10
	MOVQ grads+96(FP), R11
	MOVQ out_len+8(FP), CX

	VBROADCASTSD rho+120(FP), Y8
	VBROADCASTSD oneMinusRho+128(FP), Y9
	VBROADCASTSD eps+136(FP), Y10
	VBROADCASTSD wd+144(FP), Y11

	CMPQ CX, $4
	JL   ad_avx2_tail
ad_avx2_loop:
	VMOVUPD (R8), Y0                          // eg2
	VMOVUPD (R9), Y1                          // edp2
	VMOVUPD (R10), Y2                         // params
	VMOVUPD (R11), Y3                         // grads

	// geff = grads + wd*params
	VMOVAPD     Y3, Y4
	VFMADD231PD Y11, Y2, Y4

	// eg2 = ρ*eg2 + (1-ρ)*geff²
	VMULPD      Y4, Y4, Y5
	VMULPD      Y8, Y0, Y0
	VFMADD231PD Y9, Y5, Y0
	VMOVUPD     Y0, (R8)

	// numer = sqrt(edp2 + eps); denom = sqrt(eg2 + eps)
	VADDPD  Y10, Y1, Y6
	VSQRTPD Y6, Y6
	VADDPD  Y10, Y0, Y7
	VSQRTPD Y7, Y7

	// delta = -(numer/denom)*geff
	VDIVPD  Y7, Y6, Y12
	VMULPD  Y4, Y12, Y12
	VXORPD  Y13, Y13, Y13
	VSUBPD  Y12, Y13, Y12                     // -delta

	// edp2 = ρ*edp2 + (1-ρ)*delta²
	VMULPD      Y12, Y12, Y14
	VMULPD      Y8, Y1, Y1
	VFMADD231PD Y9, Y14, Y1
	VMOVUPD     Y1, (R9)

	// out = params + delta
	VADDPD  Y12, Y2, Y2
	VMOVUPD Y2, (AX)

	ADDQ $32, AX
	ADDQ $32, R8
	ADDQ $32, R9
	ADDQ $32, R10
	ADDQ $32, R11
	SUBQ $4, CX
	CMPQ CX, $4
	JGE  ad_avx2_loop

ad_avx2_tail:
	CMPQ CX, $2
	JL   ad_avx2_scalar
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVUPD (R10), X2
	MOVUPD (R11), X3

	MOVSD rho+120(FP), X8
	SHUFPD $0, X8, X8
	MOVSD oneMinusRho+128(FP), X9
	SHUFPD $0, X9, X9
	MOVSD eps+136(FP), X10
	SHUFPD $0, X10, X10
	MOVSD wd+144(FP), X11
	SHUFPD $0, X11, X11

	MOVAPD X3, X4
	MOVAPD X2, X14
	MULPD  X11, X14
	ADDPD  X14, X4

	MOVAPD X4, X5
	MULPD  X5, X5
	MULPD  X8, X0
	MOVAPD X5, X15
	MULPD  X9, X15
	ADDPD  X15, X0
	MOVUPD X0, (R8)

	MOVAPD X1, X6
	ADDPD  X10, X6
	SQRTPD X6, X6
	MOVAPD X0, X7
	ADDPD  X10, X7
	SQRTPD X7, X7

	MOVAPD X6, X12
	DIVPD  X7, X12
	MULPD  X4, X12
	XORPD  X13, X13
	SUBPD  X12, X13
	MOVAPD X13, X12

	MOVAPD X12, X14
	MULPD  X14, X14
	MULPD  X8, X1
	MOVAPD X14, X15
	MULPD  X9, X15
	ADDPD  X15, X1
	MOVUPD X1, (R9)

	ADDPD  X12, X2
	MOVUPD X2, (AX)

	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	ADDQ $16, R11
	SUBQ $2, CX

ad_avx2_scalar:
	CMPQ CX, $0
	JLE  ad_avx2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVSD (R10), X2
	MOVSD (R11), X3

	MOVSD rho+120(FP), X8
	MOVSD oneMinusRho+128(FP), X9
	MOVSD eps+136(FP), X10
	MOVSD wd+144(FP), X11

	MOVAPD X3, X4
	MOVAPD X2, X14
	MULSD  X11, X14
	ADDSD  X14, X4

	MOVAPD X4, X5
	MULSD  X5, X5
	MULSD  X8, X0
	MOVAPD X5, X15
	MULSD  X9, X15
	ADDSD  X15, X0
	MOVSD  X0, (R8)

	MOVAPD X1, X6
	ADDSD  X10, X6
	SQRTSD X6, X6
	MOVAPD X0, X7
	ADDSD  X10, X7
	SQRTSD X7, X7

	MOVAPD X6, X12
	DIVSD  X7, X12
	MULSD  X4, X12
	XORPD  X13, X13
	SUBSD  X12, X13
	MOVAPD X13, X12

	MOVAPD X12, X14
	MULSD  X14, X14
	MULSD  X8, X1
	MOVAPD X14, X15
	MULSD  X9, X15
	ADDSD  X15, X1
	MOVSD  X1, (R9)

	ADDSD  X12, X2
	MOVSD  X2, (AX)

ad_avx2_done:
	VZEROUPPER
	RET
