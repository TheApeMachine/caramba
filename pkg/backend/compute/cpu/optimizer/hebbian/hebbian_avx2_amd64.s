#include "textflag.h"

// hebbStepAVX2(out, params, grads []float64, lr float64)
//   out[i] = params[i] + lr*grads[i]
TEXT ·hebbStepAVX2(SB), NOSPLIT, $0-80
	MOVQ out+0(FP), AX
	MOVQ params+24(FP), R8
	MOVQ grads+48(FP), R9
	MOVQ out_len+8(FP), CX
	VBROADCASTSD lr+72(FP), Y8

	CMPQ CX, $4
	JL   hebb_avx2_tail
hebb_avx2_loop:
	VMOVUPD     (R8), Y0
	VMOVUPD     (R9), Y1
	VFMADD231PD Y8, Y1, Y0
	VMOVUPD     Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, R8
	ADDQ $32, R9
	SUBQ $4, CX
	CMPQ CX, $4
	JGE  hebb_avx2_loop

hebb_avx2_tail:
	CMPQ CX, $2
	JL   hebb_avx2_scalar
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVSD lr+72(FP), X8
	SHUFPD $0, X8, X8
	MULPD X8, X1
	ADDPD X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $2, CX

hebb_avx2_scalar:
	CMPQ CX, $0
	JLE hebb_avx2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVSD lr+72(FP), X8
	MULSD X8, X1
	ADDSD X1, X0
	MOVSD X0, (AX)
hebb_avx2_done:
	VZEROUPPER
	RET

TEXT ·hebbStepSSE2(SB), NOSPLIT, $0-80
	MOVQ out+0(FP), AX
	MOVQ params+24(FP), R8
	MOVQ grads+48(FP), R9
	MOVQ out_len+8(FP), CX
	MOVSD lr+72(FP), X8
	SHUFPD $0, X8, X8

	CMPQ CX, $2
	JL hebb_sse2_tail
hebb_sse2_loop:
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MULPD X8, X1
	ADDPD X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $2, CX
	CMPQ CX, $2
	JGE hebb_sse2_loop

hebb_sse2_tail:
	CMPQ CX, $0
	JLE hebb_sse2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MULSD X8, X1
	ADDSD X1, X0
	MOVSD X0, (AX)
hebb_sse2_done:
	RET

// hebbStepNormAVX2(out, params, grads []float64, lr, maxNorm float64) float64
// (returns post-update L2 norm so Go can choose scaling)
// Implements: out = params + lr*grads ; returns sqrt(sum(out²))
TEXT ·hebbStepNormAVX2(SB), NOSPLIT, $0-88
	MOVQ out+0(FP), AX
	MOVQ params+24(FP), R8
	MOVQ grads+48(FP), R9
	MOVQ out_len+8(FP), CX
	VBROADCASTSD lr+72(FP), Y8
	VXORPD Y10, Y10, Y10                       // accumulator

	MOVQ CX, R11
	CMPQ CX, $4
	JL   hebbn_avx2_tail
hebbn_avx2_loop:
	VMOVUPD     (R8), Y0
	VMOVUPD     (R9), Y1
	VFMADD231PD Y8, Y1, Y0
	VMOVUPD     Y0, (AX)
	VFMADD231PD Y0, Y0, Y10                    // sumsq += out²
	ADDQ $32, AX
	ADDQ $32, R8
	ADDQ $32, R9
	SUBQ $4, CX
	CMPQ CX, $4
	JGE  hebbn_avx2_loop

	// horizontal sum of Y10 into X10[0]
	VEXTRACTF128 $1, Y10, X11
	VADDPD       X11, X10, X10
	VHADDPD      X10, X10, X10

hebbn_avx2_tail:
	CMPQ CX, $2
	JL   hebbn_avx2_scalar
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVSD lr+72(FP), X8
	SHUFPD $0, X8, X8
	MULPD X8, X1
	ADDPD X1, X0
	MOVUPD X0, (AX)
	MOVAPD X0, X11
	MULPD X11, X11
	HADDPD X11, X11
	ADDSD X11, X10
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $2, CX

hebbn_avx2_scalar:
	CMPQ CX, $0
	JLE hebbn_avx2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVSD lr+72(FP), X8
	MULSD X8, X1
	ADDSD X1, X0
	MOVSD X0, (AX)
	MOVAPD X0, X11
	MULSD X11, X11
	ADDSD X11, X10

hebbn_avx2_done:
	SQRTSD X10, X10
	MOVSD X10, ret+80(FP)
	VZEROUPPER
	RET

TEXT ·hebbStepNormSSE2(SB), NOSPLIT, $0-88
	MOVQ out+0(FP), AX
	MOVQ params+24(FP), R8
	MOVQ grads+48(FP), R9
	MOVQ out_len+8(FP), CX
	MOVSD lr+72(FP), X8
	SHUFPD $0, X8, X8
	XORPD X10, X10

	CMPQ CX, $2
	JL hebbn_sse2_tail
hebbn_sse2_loop:
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MULPD X8, X1
	ADDPD X1, X0
	MOVUPD X0, (AX)
	MOVAPD X0, X11
	MULPD X11, X11
	ADDPD X11, X10
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $2, CX
	CMPQ CX, $2
	JGE hebbn_sse2_loop

	HADDPD X10, X10

hebbn_sse2_tail:
	CMPQ CX, $0
	JLE hebbn_sse2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MULSD X8, X1
	ADDSD X1, X0
	MOVSD X0, (AX)
	MOVAPD X0, X11
	MULSD X11, X11
	ADDSD X11, X10

hebbn_sse2_done:
	SQRTSD X10, X10
	MOVSD X10, ret+80(FP)
	RET

// hebbScaleAVX2(out []float64, scale float64) — in-place vector scale.
TEXT ·hebbScaleAVX2(SB), NOSPLIT, $0-32
	MOVQ out+0(FP), AX
	MOVQ out_len+8(FP), CX
	VBROADCASTSD scale+24(FP), Y8
	CMPQ CX, $4
	JL hsc_avx2_tail
hsc_avx2_loop:
	VMOVUPD (AX), Y0
	VMULPD Y8, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	SUBQ $4, CX
	CMPQ CX, $4
	JGE hsc_avx2_loop
hsc_avx2_tail:
	CMPQ CX, $2
	JL hsc_avx2_scalar
	MOVUPD (AX), X0
	MOVSD scale+24(FP), X8
	SHUFPD $0, X8, X8
	MULPD X8, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	SUBQ $2, CX
hsc_avx2_scalar:
	CMPQ CX, $0
	JLE hsc_avx2_done
	MOVSD (AX), X0
	MOVSD scale+24(FP), X8
	MULSD X8, X0
	MOVSD X0, (AX)
hsc_avx2_done:
	VZEROUPPER
	RET

TEXT ·hebbScaleSSE2(SB), NOSPLIT, $0-32
	MOVQ out+0(FP), AX
	MOVQ out_len+8(FP), CX
	MOVSD scale+24(FP), X8
	SHUFPD $0, X8, X8
	CMPQ CX, $2
	JL hsc_sse2_tail
hsc_sse2_loop:
	MOVUPD (AX), X0
	MULPD X8, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	SUBQ $2, CX
	CMPQ CX, $2
	JGE hsc_sse2_loop
hsc_sse2_tail:
	CMPQ CX, $0
	JLE hsc_sse2_done
	MOVSD (AX), X0
	MULSD X8, X0
	MOVSD X0, (AX)
hsc_sse2_done:
	RET

// ojaStepAVX2(out, params, grads []float64, lr, postSq float64)
//   out[i] = params[i] + lr*grads[i] - lr*postSq*params[i]
TEXT ·ojaStepAVX2(SB), NOSPLIT, $0-88
	MOVQ out+0(FP), AX
	MOVQ params+24(FP), R8
	MOVQ grads+48(FP), R9
	MOVQ out_len+8(FP), CX
	VBROADCASTSD lr+72(FP), Y8
	VBROADCASTSD postSq+80(FP), Y9
	VMULPD Y8, Y9, Y9                          // lr*postSq broadcast
	CMPQ CX, $4
	JL oja_avx2_tail
oja_avx2_loop:
	VMOVUPD (R8), Y0
	VMOVUPD (R9), Y1
	VMOVAPD Y0, Y2
	VFMADD231PD Y8, Y1, Y2                     // params + lr*grads
	VFNMADD231PD Y9, Y0, Y2                    // - lr*postSq*params
	VMOVUPD Y2, (AX)
	ADDQ $32, AX
	ADDQ $32, R8
	ADDQ $32, R9
	SUBQ $4, CX
	CMPQ CX, $4
	JGE oja_avx2_loop
oja_avx2_tail:
	CMPQ CX, $2
	JL oja_avx2_scalar
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVSD lr+72(FP), X8
	SHUFPD $0, X8, X8
	MOVSD postSq+80(FP), X9
	SHUFPD $0, X9, X9
	MULPD X8, X9
	MOVAPD X0, X2
	MOVAPD X1, X10
	MULPD X8, X10
	ADDPD X10, X2
	MOVAPD X0, X10
	MULPD X9, X10
	SUBPD X10, X2
	MOVUPD X2, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $2, CX
oja_avx2_scalar:
	CMPQ CX, $0
	JLE oja_avx2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVSD lr+72(FP), X8
	MOVSD postSq+80(FP), X9
	MULSD X8, X9
	MOVAPD X0, X2
	MOVAPD X1, X10
	MULSD X8, X10
	ADDSD X10, X2
	MOVAPD X0, X10
	MULSD X9, X10
	SUBSD X10, X2
	MOVSD X2, (AX)
oja_avx2_done:
	VZEROUPPER
	RET

TEXT ·ojaStepSSE2(SB), NOSPLIT, $0-88
	MOVQ out+0(FP), AX
	MOVQ params+24(FP), R8
	MOVQ grads+48(FP), R9
	MOVQ out_len+8(FP), CX
	MOVSD lr+72(FP), X8
	SHUFPD $0, X8, X8
	MOVSD postSq+80(FP), X9
	SHUFPD $0, X9, X9
	MULPD X8, X9
	CMPQ CX, $2
	JL oja_sse2_tail
oja_sse2_loop:
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVAPD X0, X2
	MOVAPD X1, X10
	MULPD X8, X10
	ADDPD X10, X2
	MOVAPD X0, X10
	MULPD X9, X10
	SUBPD X10, X2
	MOVUPD X2, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $2, CX
	CMPQ CX, $2
	JGE oja_sse2_loop
oja_sse2_tail:
	CMPQ CX, $0
	JLE oja_sse2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVAPD X0, X2
	MOVAPD X1, X10
	MULSD X8, X10
	ADDSD X10, X2
	MOVAPD X0, X10
	MULSD X9, X10
	SUBSD X10, X2
	MOVSD X2, (AX)
oja_sse2_done:
	RET

// reduceSumSqAVX2(a []float64) float64
TEXT ·reduceSumSqAVX2(SB), NOSPLIT, $0-32
	MOVQ a+0(FP), AX
	MOVQ a_len+8(FP), CX
	VXORPD Y0, Y0, Y0
	CMPQ CX, $4
	JL rss_avx2_tail
rss_avx2_loop:
	VMOVUPD (AX), Y1
	VFMADD231PD Y1, Y1, Y0
	ADDQ $32, AX
	SUBQ $4, CX
	CMPQ CX, $4
	JGE rss_avx2_loop
	VEXTRACTF128 $1, Y0, X2
	VADDPD X2, X0, X0
	VHADDPD X0, X0, X0
rss_avx2_tail:
	CMPQ CX, $2
	JL rss_avx2_scalar
	MOVUPD (AX), X1
	MULPD X1, X1
	HADDPD X1, X1
	ADDSD X1, X0
	ADDQ $16, AX
	SUBQ $2, CX
rss_avx2_scalar:
	CMPQ CX, $0
	JLE rss_avx2_done
	MOVSD (AX), X1
	MULSD X1, X1
	ADDSD X1, X0
rss_avx2_done:
	MOVSD X0, ret+24(FP)
	VZEROUPPER
	RET

TEXT ·reduceSumSqSSE2(SB), NOSPLIT, $0-32
	MOVQ a+0(FP), AX
	MOVQ a_len+8(FP), CX
	XORPD X0, X0
	CMPQ CX, $2
	JL rss_sse2_tail
rss_sse2_loop:
	MOVUPD (AX), X1
	MULPD X1, X1
	ADDPD X1, X0
	ADDQ $16, AX
	SUBQ $2, CX
	CMPQ CX, $2
	JGE rss_sse2_loop
	HADDPD X0, X0
rss_sse2_tail:
	CMPQ CX, $0
	JLE rss_sse2_done
	MOVSD (AX), X1
	MULSD X1, X1
	ADDSD X1, X0
rss_sse2_done:
	MOVSD X0, ret+24(FP)
	RET
