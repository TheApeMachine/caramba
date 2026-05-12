#include "textflag.h"

// adamStepAVX2(out, mState, vState, params, grads []float64,
//              beta1, oneMinusBeta1, beta2, oneMinusBeta2,
//              lrT, eps, lrWD float64)
//
// Fused per-element Adam update for AVX2+FMA:
//   m[i] = β1*m[i] + (1-β1)*g[i]
//   v[i] = β2*v[i] + (1-β2)*g[i]^2
//   denom = sqrt(v[i]) + ε
//   p_wd  = params[i] * (1 - lrWD)       (AdamW: lrWD = lr*wd)
//   out[i] = p_wd - lrT * m[i] / denom
//
// All scalars must be pre-broadcast; the kernel reads 4 lanes per iteration
// in the AVX2 body and 2 lanes per iteration in the trailing SSE2 body so
// every element is processed by a SIMD instruction.
//
// ABI0 frame:
//   out+0       (3 words: ptr,len,cap)
//   m+24
//   v+48
//   params+72
//   grads+96
//   beta1+120 / oneMinusBeta1+128 / beta2+136 / oneMinusBeta2+144
//   lrT+152 / eps+160 / lrWD+168

TEXT ·adamStepAVX2(SB), NOSPLIT, $0-176
	MOVQ out+0(FP), AX
	MOVQ m+24(FP), R8
	MOVQ v+48(FP), R9
	MOVQ params+72(FP), R10
	MOVQ grads+96(FP), R11
	MOVQ out_len+8(FP), CX

	VBROADCASTSD beta1+120(FP), Y8
	VBROADCASTSD oneMinusBeta1+128(FP), Y9
	VBROADCASTSD beta2+136(FP), Y10
	VBROADCASTSD oneMinusBeta2+144(FP), Y11
	VBROADCASTSD lrT+152(FP), Y12
	VBROADCASTSD eps+160(FP), Y13
	VBROADCASTSD lrWD+168(FP), Y14

	CMPQ CX, $4
	JL   adam_avx2_tail
adam_avx2_loop:
	VMOVUPD (R8), Y0                          // m
	VMOVUPD (R9), Y1                          // v
	VMOVUPD (R10), Y2                         // params
	VMOVUPD (R11), Y3                         // grads

	// g2 = g*g
	VMULPD  Y3, Y3, Y4

	// m = β1*m + (1-β1)*g
	VMULPD       Y8, Y0, Y0
	VFMADD231PD  Y9, Y3, Y0
	VMOVUPD      Y0, (R8)

	// v = β2*v + (1-β2)*g2
	VMULPD       Y10, Y1, Y1
	VFMADD231PD  Y11, Y4, Y1
	VMOVUPD      Y1, (R9)

	// denom = sqrt(v) + eps
	VSQRTPD Y1, Y5
	VADDPD  Y13, Y5, Y5

	// upd = m / denom
	VDIVPD Y5, Y0, Y6

	// out = params - lrWD*params - lrT*upd  =  params*(1-lrWD) - lrT*upd
	// First: tmp = params - lrWD*params  using FMA: tmp = params - lrWD*params
	VMOVAPD      Y2, Y7
	VFNMADD231PD Y14, Y2, Y7                  // Y7 -= lrWD*Y2
	// Then: Y7 -= lrT * Y6
	VFNMADD231PD Y12, Y6, Y7

	VMOVUPD Y7, (AX)

	ADDQ $32, AX
	ADDQ $32, R8
	ADDQ $32, R9
	ADDQ $32, R10
	ADDQ $32, R11
	SUBQ $4, CX
	CMPQ CX, $4
	JGE  adam_avx2_loop

adam_avx2_tail:
	// SSE2 2-wide for the next pair
	CMPQ CX, $2
	JL   adam_avx2_scalar
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVUPD (R10), X2
	MOVUPD (R11), X3

	MOVAPD X3, X4
	MULPD  X3, X4                              // g2

	MOVSD  beta1+120(FP), X8
	SHUFPD $0, X8, X8
	MULPD  X8, X0                              // β1*m
	MOVSD  oneMinusBeta1+128(FP), X9
	SHUFPD $0, X9, X9
	MOVAPD X3, X5
	MULPD  X9, X5
	ADDPD  X5, X0                              // m
	MOVUPD X0, (R8)

	MOVSD  beta2+136(FP), X10
	SHUFPD $0, X10, X10
	MULPD  X10, X1                             // β2*v
	MOVSD  oneMinusBeta2+144(FP), X11
	SHUFPD $0, X11, X11
	MOVAPD X4, X5
	MULPD  X11, X5
	ADDPD  X5, X1                              // v
	MOVUPD X1, (R9)

	MOVAPD X1, X5
	SQRTPD X5, X5
	MOVSD  eps+160(FP), X13
	SHUFPD $0, X13, X13
	ADDPD  X13, X5                             // denom

	MOVAPD X0, X6
	DIVPD  X5, X6                              // upd

	MOVSD  lrWD+168(FP), X14
	SHUFPD $0, X14, X14
	MOVAPD X2, X7
	MOVAPD X2, X15
	MULPD  X14, X15
	SUBPD  X15, X7                             // p - lrWD*p

	MOVSD  lrT+152(FP), X12
	SHUFPD $0, X12, X12
	MULPD  X12, X6
	SUBPD  X6, X7                              // -= lrT*upd

	MOVUPD X7, (AX)

	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	ADDQ $16, R11
	SUBQ $2, CX

adam_avx2_scalar:
	// Final lane (1 element remaining when CX==1)
	CMPQ CX, $0
	JLE  adam_avx2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVSD (R10), X2
	MOVSD (R11), X3

	MOVAPD X3, X4
	MULSD  X3, X4

	MOVSD beta1+120(FP), X8
	MULSD X8, X0
	MOVSD oneMinusBeta1+128(FP), X9
	MOVAPD X3, X5
	MULSD X9, X5
	ADDSD X5, X0
	MOVSD X0, (R8)

	MOVSD beta2+136(FP), X10
	MULSD X10, X1
	MOVSD oneMinusBeta2+144(FP), X11
	MOVAPD X4, X5
	MULSD X11, X5
	ADDSD X5, X1
	MOVSD X1, (R9)

	MOVAPD X1, X5
	SQRTSD X5, X5
	MOVSD eps+160(FP), X13
	ADDSD X13, X5

	MOVAPD X0, X6
	DIVSD X5, X6

	MOVSD lrWD+168(FP), X14
	MOVAPD X2, X7
	MOVAPD X2, X15
	MULSD X14, X15
	SUBSD X15, X7

	MOVSD lrT+152(FP), X12
	MULSD X12, X6
	SUBSD X6, X7

	MOVSD X7, (AX)

adam_avx2_done:
	VZEROUPPER
	RET

// adamStepSSE2 — pure SSE2 (no FMA, no AVX). Same signature, 2 lanes/iter.
TEXT ·adamStepSSE2(SB), NOSPLIT, $0-176
	MOVQ out+0(FP), AX
	MOVQ m+24(FP), R8
	MOVQ v+48(FP), R9
	MOVQ params+72(FP), R10
	MOVQ grads+96(FP), R11
	MOVQ out_len+8(FP), CX

	MOVSD  beta1+120(FP), X8
	SHUFPD $0, X8, X8
	MOVSD  oneMinusBeta1+128(FP), X9
	SHUFPD $0, X9, X9
	MOVSD  beta2+136(FP), X10
	SHUFPD $0, X10, X10
	MOVSD  oneMinusBeta2+144(FP), X11
	SHUFPD $0, X11, X11
	MOVSD  lrT+152(FP), X12
	SHUFPD $0, X12, X12
	MOVSD  eps+160(FP), X13
	SHUFPD $0, X13, X13
	MOVSD  lrWD+168(FP), X14
	SHUFPD $0, X14, X14

	CMPQ CX, $2
	JL   adam_sse2_tail
adam_sse2_loop:
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVUPD (R10), X2
	MOVUPD (R11), X3

	MOVAPD X3, X4
	MULPD  X3, X4                              // g2

	MULPD  X8, X0
	MOVAPD X3, X5
	MULPD  X9, X5
	ADDPD  X5, X0
	MOVUPD X0, (R8)

	MULPD  X10, X1
	MOVAPD X4, X5
	MULPD  X11, X5
	ADDPD  X5, X1
	MOVUPD X1, (R9)

	MOVAPD X1, X5
	SQRTPD X5, X5
	ADDPD  X13, X5

	MOVAPD X0, X6
	DIVPD  X5, X6

	MOVAPD X2, X7
	MOVAPD X2, X15
	MULPD  X14, X15
	SUBPD  X15, X7

	MULPD X12, X6
	SUBPD X6, X7
	MOVUPD X7, (AX)

	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	ADDQ $16, R11
	SUBQ $2, CX
	CMPQ CX, $2
	JGE  adam_sse2_loop

adam_sse2_tail:
	CMPQ CX, $0
	JLE  adam_sse2_done

	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVSD (R10), X2
	MOVSD (R11), X3

	MOVAPD X3, X4
	MULSD X3, X4

	MULSD X8, X0
	MOVAPD X3, X5
	MULSD X9, X5
	ADDSD X5, X0
	MOVSD X0, (R8)

	MULSD X10, X1
	MOVAPD X4, X5
	MULSD X11, X5
	ADDSD X5, X1
	MOVSD X1, (R9)

	MOVAPD X1, X5
	SQRTSD X5, X5
	ADDSD X13, X5

	MOVAPD X0, X6
	DIVSD X5, X6

	MOVAPD X2, X7
	MOVAPD X2, X15
	MULSD X14, X15
	SUBSD X15, X7

	MULSD X12, X6
	SUBSD X6, X7
	MOVSD X7, (AX)

adam_sse2_done:
	RET
