#include "textflag.h"

// sgdVanillaAVX2(out, params, grads []float64, lr, wd float64)
//   out[i] = params[i] - lr*(grads[i] + wd*params[i])
TEXT ·sgdVanillaAVX2(SB), NOSPLIT, $0-88
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

// sgdMomentumAVX2(out, params, grads, velocity []float64,
//                 lr, wd, momentum float64, nesterov uint64)
//   geff = grads[i] + wd*params[i]
//   v[i] = μ*v[i] + geff
//   out[i] = params[i] - lr*(nesterov ? geff + μ*v[i] : v[i])
TEXT ·sgdMomentumAVX2(SB), NOSPLIT, $0-128
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

	VMOVAPD      Y1, Y3
	VFMADD231PD  Y9, Y0, Y3
	VMULPD       Y10, Y2, Y2
	VADDPD       Y3, Y2, Y2
	VMOVUPD      Y2, (R10)

	CMPQ DX, $0
	JE   sgdm_avx2_addV
	VMULPD       Y10, Y2, Y5
	VADDPD       Y3, Y5, Y5
	JMP          sgdm_avx2_store
sgdm_avx2_addV:
	VMOVAPD Y2, Y5
sgdm_avx2_store:
	VMULPD  Y8, Y5, Y5
	VSUBPD  Y5, Y0, Y0
	VMOVUPD Y0, (AX)

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

	MOVAPD X1, X3
	MOVAPD X0, X4
	MULPD X9, X4
	ADDPD X4, X3
	MULPD X10, X2
	ADDPD X3, X2
	MOVUPD X2, (R10)

	CMPQ DX, $0
	JE sgdm_avx2_tailAddV
	MOVAPD X2, X5
	MULPD X10, X5
	ADDPD X3, X5
	JMP sgdm_avx2_tailStore
sgdm_avx2_tailAddV:
	MOVAPD X2, X5
sgdm_avx2_tailStore:
	MULPD X8, X5
	SUBPD X5, X0
	MOVUPD X0, (AX)
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

	MOVAPD X1, X3
	MOVAPD X0, X4
	MULSD X9, X4
	ADDSD X4, X3
	MULSD X10, X2
	ADDSD X3, X2
	MOVSD X2, (R10)

	CMPQ DX, $0
	JE sgdm_avx2_scalarAddV
	MOVAPD X2, X5
	MULSD X10, X5
	ADDSD X3, X5
	JMP sgdm_avx2_scalarStore
sgdm_avx2_scalarAddV:
	MOVAPD X2, X5
sgdm_avx2_scalarStore:
	MULSD X8, X5
	SUBSD X5, X0
	MOVSD X0, (AX)

sgdm_avx2_done:
	VZEROUPPER
	RET
