#include "textflag.h"

// bundleAccumAVX2(dst, src []float64)
//   dst[i] += src[i]
TEXT ·bundleAccumAVX2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), DI
	MOVQ dst_len+8(FP), CX
	CMPQ CX, $4
	JL ba_tail
ba_loop:
	VMOVUPD (AX), Y0
	VMOVUPD (DI), Y1
	VADDPD Y1, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, DI
	SUBQ $4, CX
	CMPQ CX, $4
	JGE ba_loop
ba_tail:
	CMPQ CX, $2
	JL ba_scalar
	MOVUPD (AX), X0
	MOVUPD (DI), X1
	ADDPD X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, CX
ba_scalar:
	CMPQ CX, $0
	JLE ba_done
	MOVSD (AX), X0
	MOVSD (DI), X1
	ADDSD X1, X0
	MOVSD X0, (AX)
ba_done:
	VZEROUPPER
	RET

// bundleNormalizeAVX2(dst []float64, eps float64)
//   Computes sumsq=Σ dst[i]², norm=sqrt(sumsq). If norm > eps, scales dst by 1/norm.
TEXT ·bundleNormalizeAVX2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), AX
	MOVQ dst_len+8(FP), CX
	MOVSD eps+24(FP), X10
	MOVQ AX, R8                                // saved ptr
	MOVQ CX, R9                                // saved len

	VXORPD Y0, Y0, Y0
	CMPQ CX, $4
	JL bn_ss_tail
bn_ss_loop:
	VMOVUPD (AX), Y1
	VFMADD231PD Y1, Y1, Y0
	ADDQ $32, AX
	SUBQ $4, CX
	CMPQ CX, $4
	JGE bn_ss_loop
	VEXTRACTF128 $1, Y0, X2
	VADDPD X2, X0, X0
	VHADDPD X0, X0, X0
bn_ss_tail:
	CMPQ CX, $2
	JL bn_ss_scalar
	MOVUPD (AX), X1
	MULPD X1, X1
	HADDPD X1, X1
	ADDSD X1, X0
	ADDQ $16, AX
	SUBQ $2, CX
bn_ss_scalar:
	CMPQ CX, $0
	JLE bn_have_sumsq
	MOVSD (AX), X1
	MULSD X1, X1
	ADDSD X1, X0
bn_have_sumsq:
	SQRTSD X0, X0
	UCOMISD X10, X0
	JBE bn_done                                // norm <= eps → skip
	MOVSD $1.0, X3
	DIVSD X0, X3
	VBROADCASTSD X3, Y3
	MOVAPD X3, X11

	MOVQ R8, AX
	MOVQ R9, CX
	CMPQ CX, $4
	JL bn_scale_tail
bn_scale_loop:
	VMOVUPD (AX), Y0
	VMULPD Y3, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	SUBQ $4, CX
	CMPQ CX, $4
	JGE bn_scale_loop
bn_scale_tail:
	CMPQ CX, $2
	JL bn_scale_scalar
	MOVUPD (AX), X0
	MOVAPD X11, X1
	SHUFPD $0, X1, X1
	MULPD X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	SUBQ $2, CX
bn_scale_scalar:
	CMPQ CX, $0
	JLE bn_done
	MOVSD (AX), X0
	MULSD X11, X0
	MOVSD X0, (AX)
bn_done:
	VZEROUPPER
	RET
