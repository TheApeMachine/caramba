#include "textflag.h"

// adagradStepAVX2(out, G, params, grads []float64, lr, eps, wd float64)
//   geff = grads + wd*params                (if wd != 0)
//   G[i] += geff[i]^2
//   out[i] = params[i] - lr * geff[i] / (sqrt(G[i]) + eps)
TEXT ·adagradStepAVX2(SB), NOSPLIT, $0-128
	MOVQ out+0(FP), AX
	MOVQ G+24(FP), R8
	MOVQ params+48(FP), R9
	MOVQ grads+72(FP), R10
	MOVQ out_len+8(FP), CX

	VBROADCASTSD lr+96(FP), Y8
	VBROADCASTSD eps+104(FP), Y9
	VBROADCASTSD wd+112(FP), Y10

	CMPQ CX, $4
	JL   ag_avx2_tail
ag_avx2_loop:
	VMOVUPD (R8), Y0                          // G
	VMOVUPD (R9), Y1                          // params
	VMOVUPD (R10), Y2                         // grads

	// geff = grads + wd*params
	VMOVAPD     Y2, Y3
	VFMADD231PD Y10, Y1, Y3

	// G += geff^2
	VFMADD231PD Y3, Y3, Y0
	VMOVUPD     Y0, (R8)

	// denom = sqrt(G) + eps
	VSQRTPD Y0, Y4
	VADDPD  Y9, Y4, Y4

	// upd = geff / denom
	VDIVPD Y4, Y3, Y5
	// out = params - lr*upd
	VFNMADD231PD Y8, Y5, Y1
	VMOVUPD      Y1, (AX)

	ADDQ $32, AX
	ADDQ $32, R8
	ADDQ $32, R9
	ADDQ $32, R10
	SUBQ $4, CX
	CMPQ CX, $4
	JGE  ag_avx2_loop

ag_avx2_tail:
	CMPQ CX, $2
	JL   ag_avx2_scalar

	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVUPD (R10), X2

	MOVSD lr+96(FP), X8
	SHUFPD $0, X8, X8
	MOVSD eps+104(FP), X9
	SHUFPD $0, X9, X9
	MOVSD wd+112(FP), X10
	SHUFPD $0, X10, X10

	MOVAPD X2, X3
	MOVAPD X1, X11
	MULPD  X10, X11
	ADDPD  X11, X3                            // geff

	MOVAPD X3, X11
	MULPD  X11, X11
	ADDPD  X11, X0                            // G += geff^2
	MOVUPD X0, (R8)

	MOVAPD X0, X4
	SQRTPD X4, X4
	ADDPD  X9, X4
	MOVAPD X3, X5
	DIVPD  X4, X5
	MULPD  X8, X5
	SUBPD  X5, X1
	MOVUPD X1, (AX)

	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $2, CX

ag_avx2_scalar:
	CMPQ CX, $0
	JLE  ag_avx2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVSD (R10), X2

	MOVSD lr+96(FP), X8
	MOVSD eps+104(FP), X9
	MOVSD wd+112(FP), X10

	MOVAPD X2, X3
	MOVAPD X1, X11
	MULSD  X10, X11
	ADDSD  X11, X3

	MOVAPD X3, X11
	MULSD  X11, X11
	ADDSD  X11, X0
	MOVSD  X0, (R8)

	MOVAPD X0, X4
	SQRTSD X4, X4
	ADDSD  X9, X4
	MOVAPD X3, X5
	DIVSD  X4, X5
	MULSD  X8, X5
	SUBSD  X5, X1
	MOVSD  X1, (AX)

ag_avx2_done:
	VZEROUPPER
	RET

// adagradStepSSE2 — SSE2 only.
TEXT ·adagradStepSSE2(SB), NOSPLIT, $0-128
	MOVQ out+0(FP), AX
	MOVQ G+24(FP), R8
	MOVQ params+48(FP), R9
	MOVQ grads+72(FP), R10
	MOVQ out_len+8(FP), CX

	MOVSD lr+96(FP), X8
	SHUFPD $0, X8, X8
	MOVSD eps+104(FP), X9
	SHUFPD $0, X9, X9
	MOVSD wd+112(FP), X10
	SHUFPD $0, X10, X10

	CMPQ CX, $2
	JL   ag_sse2_tail
ag_sse2_loop:
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVUPD (R10), X2

	MOVAPD X2, X3
	MOVAPD X1, X11
	MULPD  X10, X11
	ADDPD  X11, X3

	MOVAPD X3, X11
	MULPD  X11, X11
	ADDPD  X11, X0
	MOVUPD X0, (R8)

	MOVAPD X0, X4
	SQRTPD X4, X4
	ADDPD  X9, X4
	MOVAPD X3, X5
	DIVPD  X4, X5
	MULPD  X8, X5
	SUBPD  X5, X1
	MOVUPD X1, (AX)

	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $2, CX
	CMPQ CX, $2
	JGE  ag_sse2_loop

ag_sse2_tail:
	CMPQ CX, $0
	JLE  ag_sse2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVSD (R10), X2

	MOVAPD X2, X3
	MOVAPD X1, X11
	MULSD  X10, X11
	ADDSD  X11, X3

	MOVAPD X3, X11
	MULSD  X11, X11
	ADDSD  X11, X0
	MOVSD  X0, (R8)

	MOVAPD X0, X4
	SQRTSD X4, X4
	ADDSD  X9, X4
	MOVAPD X3, X5
	DIVSD  X4, X5
	MULSD  X8, X5
	SUBSD  X5, X1
	MOVSD  X1, (AX)

ag_sse2_done:
	RET
