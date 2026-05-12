#include "textflag.h"

// lbfgsSubAVX2(dst, a, b []float64)   dst[i] = a[i] - b[i]
TEXT ·lbfgsSubAVX2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ a+24(FP), R8
	MOVQ b+48(FP), R9
	MOVQ dst_len+8(FP), CX
	CMPQ CX, $4
	JL sub_avx2_tail
sub_avx2_loop:
	VMOVUPD (R8), Y0
	VMOVUPD (R9), Y1
	VSUBPD Y1, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, R8
	ADDQ $32, R9
	SUBQ $4, CX
	CMPQ CX, $4
	JGE sub_avx2_loop
sub_avx2_tail:
	CMPQ CX, $2
	JL sub_avx2_scalar
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	SUBPD X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $2, CX
sub_avx2_scalar:
	CMPQ CX, $0
	JLE sub_avx2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	SUBSD X1, X0
	MOVSD X0, (AX)
sub_avx2_done:
	VZEROUPPER
	RET

TEXT ·lbfgsSubSSE2(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), AX
	MOVQ a+24(FP), R8
	MOVQ b+48(FP), R9
	MOVQ dst_len+8(FP), CX
	CMPQ CX, $2
	JL sub_sse2_tail
sub_sse2_loop:
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	SUBPD X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $2, CX
	CMPQ CX, $2
	JGE sub_sse2_loop
sub_sse2_tail:
	CMPQ CX, $0
	JLE sub_sse2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	SUBSD X1, X0
	MOVSD X0, (AX)
sub_sse2_done:
	RET

// lbfgsDotAVX2(a, b []float64) float64
TEXT ·lbfgsDotAVX2(SB), NOSPLIT, $0-56
	MOVQ a+0(FP), AX
	MOVQ b+24(FP), R8
	MOVQ a_len+8(FP), CX
	VXORPD Y0, Y0, Y0
	CMPQ CX, $4
	JL dot_avx2_tail
dot_avx2_loop:
	VMOVUPD (AX), Y1
	VMOVUPD (R8), Y2
	VFMADD231PD Y1, Y2, Y0
	ADDQ $32, AX
	ADDQ $32, R8
	SUBQ $4, CX
	CMPQ CX, $4
	JGE dot_avx2_loop
	VEXTRACTF128 $1, Y0, X3
	VADDPD X3, X0, X0
	VHADDPD X0, X0, X0
dot_avx2_tail:
	CMPQ CX, $2
	JL dot_avx2_scalar
	MOVUPD (AX), X1
	MOVUPD (R8), X2
	MULPD X2, X1
	HADDPD X1, X1
	ADDSD X1, X0
	ADDQ $16, AX
	ADDQ $16, R8
	SUBQ $2, CX
dot_avx2_scalar:
	CMPQ CX, $0
	JLE dot_avx2_done
	MOVSD (AX), X1
	MOVSD (R8), X2
	MULSD X2, X1
	ADDSD X1, X0
dot_avx2_done:
	MOVSD X0, ret+48(FP)
	VZEROUPPER
	RET

TEXT ·lbfgsDotSSE2(SB), NOSPLIT, $0-56
	MOVQ a+0(FP), AX
	MOVQ b+24(FP), R8
	MOVQ a_len+8(FP), CX
	XORPD X0, X0
	CMPQ CX, $2
	JL dot_sse2_tail
dot_sse2_loop:
	MOVUPD (AX), X1
	MOVUPD (R8), X2
	MULPD X2, X1
	ADDPD X1, X0
	ADDQ $16, AX
	ADDQ $16, R8
	SUBQ $2, CX
	CMPQ CX, $2
	JGE dot_sse2_loop
	HADDPD X0, X0
dot_sse2_tail:
	CMPQ CX, $0
	JLE dot_sse2_done
	MOVSD (AX), X1
	MOVSD (R8), X2
	MULSD X2, X1
	ADDSD X1, X0
dot_sse2_done:
	MOVSD X0, ret+48(FP)
	RET

// lbfgsAddScaledAVX2(dst, src []float64, scale float64)  dst[i] += scale*src[i]
TEXT ·lbfgsAddScaledAVX2(SB), NOSPLIT, $0-56
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), R8
	MOVQ dst_len+8(FP), CX
	VBROADCASTSD scale+48(FP), Y8
	CMPQ CX, $4
	JL adsc_avx2_tail
adsc_avx2_loop:
	VMOVUPD (AX), Y0
	VMOVUPD (R8), Y1
	VFMADD231PD Y8, Y1, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, R8
	SUBQ $4, CX
	CMPQ CX, $4
	JGE adsc_avx2_loop
adsc_avx2_tail:
	CMPQ CX, $2
	JL adsc_avx2_scalar
	MOVUPD (AX), X0
	MOVUPD (R8), X1
	MOVSD scale+48(FP), X8
	SHUFPD $0, X8, X8
	MULPD X8, X1
	ADDPD X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	SUBQ $2, CX
adsc_avx2_scalar:
	CMPQ CX, $0
	JLE adsc_avx2_done
	MOVSD (AX), X0
	MOVSD (R8), X1
	MOVSD scale+48(FP), X8
	MULSD X8, X1
	ADDSD X1, X0
	MOVSD X0, (AX)
adsc_avx2_done:
	VZEROUPPER
	RET

TEXT ·lbfgsAddScaledSSE2(SB), NOSPLIT, $0-56
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), R8
	MOVQ dst_len+8(FP), CX
	MOVSD scale+48(FP), X8
	SHUFPD $0, X8, X8
	CMPQ CX, $2
	JL adsc_sse2_tail
adsc_sse2_loop:
	MOVUPD (AX), X0
	MOVUPD (R8), X1
	MULPD X8, X1
	ADDPD X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	SUBQ $2, CX
	CMPQ CX, $2
	JGE adsc_sse2_loop
adsc_sse2_tail:
	CMPQ CX, $0
	JLE adsc_sse2_done
	MOVSD (AX), X0
	MOVSD (R8), X1
	MULSD X8, X1
	ADDSD X1, X0
	MOVSD X0, (AX)
adsc_sse2_done:
	RET

// lbfgsScaleAVX2(dst []float64, scale float64)
TEXT ·lbfgsScaleAVX2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), AX
	MOVQ dst_len+8(FP), CX
	VBROADCASTSD scale+24(FP), Y8
	CMPQ CX, $4
	JL sc_avx2_tail
sc_avx2_loop:
	VMOVUPD (AX), Y0
	VMULPD Y8, Y0, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	SUBQ $4, CX
	CMPQ CX, $4
	JGE sc_avx2_loop
sc_avx2_tail:
	CMPQ CX, $2
	JL sc_avx2_scalar
	MOVUPD (AX), X0
	MOVSD scale+24(FP), X8
	SHUFPD $0, X8, X8
	MULPD X8, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	SUBQ $2, CX
sc_avx2_scalar:
	CMPQ CX, $0
	JLE sc_avx2_done
	MOVSD (AX), X0
	MOVSD scale+24(FP), X8
	MULSD X8, X0
	MOVSD X0, (AX)
sc_avx2_done:
	VZEROUPPER
	RET

TEXT ·lbfgsScaleSSE2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), AX
	MOVQ dst_len+8(FP), CX
	MOVSD scale+24(FP), X8
	SHUFPD $0, X8, X8
	CMPQ CX, $2
	JL sc_sse2_tail
sc_sse2_loop:
	MOVUPD (AX), X0
	MULPD X8, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	SUBQ $2, CX
	CMPQ CX, $2
	JGE sc_sse2_loop
sc_sse2_tail:
	CMPQ CX, $0
	JLE sc_sse2_done
	MOVSD (AX), X0
	MULSD X8, X0
	MOVSD X0, (AX)
sc_sse2_done:
	RET

// lbfgsParamStepAVX2(out, params, dir []float64, lr float64)
//   out[i] = params[i] - lr * dir[i]
TEXT ·lbfgsParamStepAVX2(SB), NOSPLIT, $0-80
	MOVQ out+0(FP), AX
	MOVQ params+24(FP), R8
	MOVQ dir+48(FP), R9
	MOVQ out_len+8(FP), CX
	VBROADCASTSD lr+72(FP), Y8
	CMPQ CX, $4
	JL ps_avx2_tail
ps_avx2_loop:
	VMOVUPD (R8), Y0
	VMOVUPD (R9), Y1
	VFNMADD231PD Y8, Y1, Y0
	VMOVUPD Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, R8
	ADDQ $32, R9
	SUBQ $4, CX
	CMPQ CX, $4
	JGE ps_avx2_loop
ps_avx2_tail:
	CMPQ CX, $2
	JL ps_avx2_scalar
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MOVSD lr+72(FP), X8
	SHUFPD $0, X8, X8
	MULPD X8, X1
	SUBPD X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $2, CX
ps_avx2_scalar:
	CMPQ CX, $0
	JLE ps_avx2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MOVSD lr+72(FP), X8
	MULSD X8, X1
	SUBSD X1, X0
	MOVSD X0, (AX)
ps_avx2_done:
	VZEROUPPER
	RET

TEXT ·lbfgsParamStepSSE2(SB), NOSPLIT, $0-80
	MOVQ out+0(FP), AX
	MOVQ params+24(FP), R8
	MOVQ dir+48(FP), R9
	MOVQ out_len+8(FP), CX
	MOVSD lr+72(FP), X8
	SHUFPD $0, X8, X8
	CMPQ CX, $2
	JL ps_sse2_tail
ps_sse2_loop:
	MOVUPD (R8), X0
	MOVUPD (R9), X1
	MULPD X8, X1
	SUBPD X1, X0
	MOVUPD X0, (AX)
	ADDQ $16, AX
	ADDQ $16, R8
	ADDQ $16, R9
	SUBQ $2, CX
	CMPQ CX, $2
	JGE ps_sse2_loop
ps_sse2_tail:
	CMPQ CX, $0
	JLE ps_sse2_done
	MOVSD (R8), X0
	MOVSD (R9), X1
	MULSD X8, X1
	SUBSD X1, X0
	MOVSD X0, (AX)
ps_sse2_done:
	RET
