#include "textflag.h"

// sgdVanillaAVX2(out, params, grads []float64, lr, wd float64)
//   out[i] = params[i] - lr*(grads[i] + wd*params[i])
TEXT ·sgdVanillaAVX2(SB), NOSPLIT, $0-104
	MOVQ out+0(FP), AX
	MOVQ params+24(FP), R8
	MOVQ grads+48(FP), R9
	MOVQ out_len+8(FP), CX
	VBROADCASTSD lr+72(FP), Y8
	VBROADCASTSD wd+80(FP), Y9

	CMPQ CX, $4
	JL   sgdv_avx2_tail
sgdv_avx2_loop:
	VMOVUPD (R8), Y0
	VMOVUPD (R9), Y1
	// geff = g + wd*p
	VFMADD231PD Y9, Y0, Y1
	// out = p - lr*geff
	VFNMADD231PD Y8, Y1, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, R8
	ADDQ $32, R9
	SUBQ $4, CX
	CMPQ CX, $4
	JGE  sgdv_avx2_loop

sgdv_avx2_tail:
	CMPQ CX, $2
	JL   sgdv_avx2_scalar
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVSD lr+72(FP), X8
	SHUFPD $0, X8, X8
	MOVSD wd+80(FP), X9
	SHUFPD $0, X9, X9
	MOVAPD X0, X2
	MULPD X9, X2
	ADDPD X2, X1
	MULPD X8, X1
	SUBPD X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $2, CX

sgdv_avx2_scalar:
	CMPQ CX, $0
	JLE  sgdv_avx2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVSD lr+72(FP), X8
	MOVSD wd+80(FP), X9
	MOVAPD X0, X2
	MULSD X9, X2
	ADDSD X2, X1
	MULSD X8, X1
	SUBSD X1, X0
	MOVSD X0, (AX)
sgdv_avx2_done:
	VZEROUPPER
	RET

TEXT ·sgdVanillaSSE2(SB), NOSPLIT, $0-104
	MOVQ out+0(FP), AX
	MOVQ params+24(FP), R8
	MOVQ grads+48(FP), R9
	MOVQ out_len+8(FP), CX
	MOVSD lr+72(FP), X8
	SHUFPD $0, X8, X8
	MOVSD wd+80(FP), X9
	SHUFPD $0, X9, X9

	CMPQ CX, $2
	JL   sgdv_sse2_tail
sgdv_sse2_loop:
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVAPD X0, X2
	MULPD X9, X2
	ADDPD X2, X1
	MULPD X8, X1
	SUBPD X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $2, CX
	CMPQ CX, $2
	JGE  sgdv_sse2_loop

sgdv_sse2_tail:
	CMPQ CX, $0
	JLE  sgdv_sse2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVAPD X0, X2
	MULSD X9, X2
	ADDSD X2, X1
	MULSD X8, X1
	SUBSD X1, X0
	MOVSD X0, (AX)
sgdv_sse2_done:
	RET

// sgdMomentumAVX2(out, params, grads, velocity []float64,
//                 lr, wd, momentum float64, nesterov uint64)
//   v[i] = μ*v[i] - lr*grads[i]
//   out[i] = params[i] - lr*wd*params[i] + (nesterov ? μ*v[i] - lr*g[i] : v[i])
TEXT ·sgdMomentumAVX2(SB), NOSPLIT, $0-136
	MOVQ out+0(FP), AX
	MOVQ params+24(FP), R8
	MOVQ grads+48(FP), R9
	MOVQ velocity+72(FP), R10
	MOVQ out_len+8(FP), CX
	VBROADCASTSD lr+96(FP), Y8
	VBROADCASTSD wd+104(FP), Y9
	VBROADCASTSD momentum+112(FP), Y10
	MOVQ nesterov+120(FP), DX

	CMPQ CX, $4
	JL   sgdm_avx2_tail
sgdm_avx2_loop:
	VMOVUPD (R8), Y0                            // params
	VMOVUPD (R9), Y1                            // grads
	VMOVUPD (R10), Y2                           // velocity

	// v = μ*v - lr*g
	VMULPD       Y10, Y2, Y2
	VFNMADD231PD Y8, Y1, Y2
	VMOVUPD      Y2, (R10)

	// out = p - lr*wd*p   (start from params, subtract scaled p)
	VMOVAPD      Y0, Y3
	VMULPD       Y9, Y0, Y4
	VFNMADD231PD Y8, Y4, Y3                     // tmp - = lr * (wd*p)
	// equivalent to: Y3 = Y0 - lr*wd*p

	// Add velocity contribution
	CMPQ DX, $0
	JE   sgdm_avx2_addV
	// Nesterov: out += μ*v - lr*g
	VMULPD       Y10, Y2, Y5
	VFNMADD231PD Y8, Y1, Y5
	VADDPD       Y5, Y3, Y3
	JMP          sgdm_avx2_store
sgdm_avx2_addV:
	VADDPD Y2, Y3, Y3
sgdm_avx2_store:
	VMOVUPD Y3, (AX)

	ADDQ $32, AX
	ADDQ $32, R8
	ADDQ $32, R9
	ADDQ $32, R10
	SUBQ $4, CX
	CMPQ CX, $4
	JGE  sgdm_avx2_loop

sgdm_avx2_tail:
	CMPQ CX, $2
	JL   sgdm_avx2_scalar
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVUPD (R10), X2
	MOVSD lr+96(FP), X8
	SHUFPD $0, X8, X8
	MOVSD wd+104(FP), X9
	SHUFPD $0, X9, X9
	MOVSD momentum+112(FP), X10
	SHUFPD $0, X10, X10

	MULPD X10, X2
	MOVAPD X1, X11
	MULPD X8, X11
	SUBPD X11, X2
	MOVUPD X2, (R10)

	MOVAPD X0, X3
	MOVAPD X0, X4
	MULPD X9, X4
	MULPD X8, X4
	SUBPD X4, X3

	CMPQ DX, $0
	JE sgdm_avx2_tailAddV
	MOVAPD X2, X5
	MULPD X10, X5
	MOVAPD X1, X11
	MULPD X8, X11
	SUBPD X11, X5
	ADDPD X5, X3
	JMP sgdm_avx2_tailStore
sgdm_avx2_tailAddV:
	ADDPD X2, X3
sgdm_avx2_tailStore:
	MOVUPD X3, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $2, CX

sgdm_avx2_scalar:
	CMPQ CX, $0
	JLE sgdm_avx2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVSD (R10), X2
	MOVSD lr+96(FP), X8
	MOVSD wd+104(FP), X9
	MOVSD momentum+112(FP), X10

	MULSD X10, X2
	MOVAPD X1, X11
	MULSD X8, X11
	SUBSD X11, X2
	MOVSD X2, (R10)

	MOVAPD X0, X3
	MOVAPD X0, X4
	MULSD X9, X4
	MULSD X8, X4
	SUBSD X4, X3
	CMPQ DX, $0
	JE sgdm_avx2_scalarAddV
	MOVAPD X2, X5
	MULSD X10, X5
	MOVAPD X1, X11
	MULSD X8, X11
	SUBSD X11, X5
	ADDSD X5, X3
	JMP sgdm_avx2_scalarStore
sgdm_avx2_scalarAddV:
	ADDSD X2, X3
sgdm_avx2_scalarStore:
	MOVSD X3, (AX)

sgdm_avx2_done:
	VZEROUPPER
	RET

TEXT ·sgdMomentumSSE2(SB), NOSPLIT, $0-136
	MOVQ out+0(FP), AX
	MOVQ params+24(FP), R8
	MOVQ grads+48(FP), R9
	MOVQ velocity+72(FP), R10
	MOVQ out_len+8(FP), CX
	MOVSD lr+96(FP), X8
	SHUFPD $0, X8, X8
	MOVSD wd+104(FP), X9
	SHUFPD $0, X9, X9
	MOVSD momentum+112(FP), X10
	SHUFPD $0, X10, X10
	MOVQ nesterov+120(FP), DX

	CMPQ CX, $2
	JL   sgdm_sse2_tail
sgdm_sse2_loop:
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVUPD (R10), X2

	MULPD X10, X2
	MOVAPD X1, X11
	MULPD X8, X11
	SUBPD X11, X2
	MOVUPD X2, (R10)

	MOVAPD X0, X3
	MOVAPD X0, X4
	MULPD X9, X4
	MULPD X8, X4
	SUBPD X4, X3
	CMPQ DX, $0
	JE sgdm_sse2_addV
	MOVAPD X2, X5
	MULPD X10, X5
	MOVAPD X1, X11
	MULPD X8, X11
	SUBPD X11, X5
	ADDPD X5, X3
	JMP sgdm_sse2_store
sgdm_sse2_addV:
	ADDPD X2, X3
sgdm_sse2_store:
	MOVUPD X3, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $2, CX
	CMPQ CX, $2
	JGE  sgdm_sse2_loop

sgdm_sse2_tail:
	CMPQ CX, $0
	JLE sgdm_sse2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVSD (R10), X2

	MULSD X10, X2
	MOVAPD X1, X11
	MULSD X8, X11
	SUBSD X11, X2
	MOVSD X2, (R10)

	MOVAPD X0, X3
	MOVAPD X0, X4
	MULSD X9, X4
	MULSD X8, X4
	SUBSD X4, X3
	CMPQ DX, $0
	JE sgdm_sse2_addVtail
	MOVAPD X2, X5
	MULSD X10, X5
	MOVAPD X1, X11
	MULSD X8, X11
	SUBSD X11, X5
	ADDSD X5, X3
	JMP sgdm_sse2_storeTail
sgdm_sse2_addVtail:
	ADDSD X2, X3
sgdm_sse2_storeTail:
	MOVSD X3, (AX)

sgdm_sse2_done:
	RET
