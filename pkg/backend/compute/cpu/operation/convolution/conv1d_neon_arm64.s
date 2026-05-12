#include "textflag.h"

// conv1dNEON — full conv1d forward pass using ARM NEON/FP.
//
// func conv1dNEON(out, x, weight, bias []float64,
//                 n, inC, l, outC, kSize, stride, pad, dilation, groups, lOut int)
//
// Arg offsets from FP (slices = 24 bytes each):
//   out:      0(FP)
//   x:       24(FP)
//   weight:  48(FP)
//   bias:    72(FP)
//   n:       96(FP)
//   inC:    104(FP)
//   l:      112(FP)
//   outC:   120(FP)
//   kSize:  128(FP)
//   stride: 136(FP)
//   pad:    144(FP)
//   dilation:152(FP)
//   groups: 160(FP)
//   lOut:   168(FP)
//
// Frame: 256 bytes for spills.
#define nN_OUTP    0
#define nN_XP      8
#define nN_WP      16
#define nN_BP      24
#define nN_N       32
#define nN_INC     40
#define nN_L       48
#define nN_OUTC    56
#define nN_KSIZE   64
#define nN_STRIDE  72
#define nN_PAD     80
#define nN_DIL     88
#define nN_GROUPS  96
#define nN_LOUT    104
#define nN_NI      112
#define nN_GRP     120
#define nN_OC      128
#define nN_LO      136
#define nN_IC      144
#define nN_ICPERG  152
#define nN_OCPERG  160
#define nN_OCSTART 168
#define nN_ICSTART 176
#define nN_WROWP   184
#define nN_SUM     192

TEXT ·conv1dNEON(SB), $256-176
	MOVD out+0(FP), R0
	MOVD R0, nN_OUTP(RSP)
	MOVD x+24(FP), R0
	MOVD R0, nN_XP(RSP)
	MOVD weight+48(FP), R0
	MOVD R0, nN_WP(RSP)
	MOVD bias+72(FP), R0
	MOVD R0, nN_BP(RSP)
	MOVD n+96(FP), R0
	MOVD R0, nN_N(RSP)
	MOVD inC+104(FP), R0
	MOVD R0, nN_INC(RSP)
	MOVD l+112(FP), R0
	MOVD R0, nN_L(RSP)
	MOVD outC+120(FP), R0
	MOVD R0, nN_OUTC(RSP)
	MOVD kSize+128(FP), R0
	MOVD R0, nN_KSIZE(RSP)
	MOVD stride+136(FP), R0
	MOVD R0, nN_STRIDE(RSP)
	MOVD pad+144(FP), R0
	MOVD R0, nN_PAD(RSP)
	MOVD dilation+152(FP), R0
	MOVD R0, nN_DIL(RSP)
	MOVD groups+160(FP), R0
	MOVD R0, nN_GROUPS(RSP)
	MOVD lOut+168(FP), R0
	MOVD R0, nN_LOUT(RSP)

	MOVD $0, R0
	MOVD R0, nN_NI(RSP)

neon1d_loop_ni:
	MOVD nN_NI(RSP), R0
	MOVD nN_N(RSP), R1
	CMP  R1, R0
	BGE  neon1d_done

	// icPerGroup = inC / groups
	MOVD nN_INC(RSP), R0
	MOVD nN_GROUPS(RSP), R1
	UDIV R1, R0, R2       // R2 = icPerGroup
	MOVD R2, nN_ICPERG(RSP)

	// ocPerGroup = outC / groups
	MOVD nN_OUTC(RSP), R0
	MOVD nN_GROUPS(RSP), R1
	UDIV R1, R0, R2
	MOVD R2, nN_OCPERG(RSP)

	MOVD $0, R0
	MOVD R0, nN_GRP(RSP)

neon1d_loop_g:
	MOVD nN_GRP(RSP), R0
	MOVD nN_GROUPS(RSP), R1
	CMP  R1, R0
	BGE  neon1d_next_ni

	// ocStart = g * ocPerGroup
	MOVD nN_GRP(RSP), R0
	MOVD nN_OCPERG(RSP), R1
	MUL  R1, R0, R2
	MOVD R2, nN_OCSTART(RSP)

	// icStart = g * icPerGroup
	MOVD nN_GRP(RSP), R0
	MOVD nN_ICPERG(RSP), R1
	MUL  R1, R0, R2
	MOVD R2, nN_ICSTART(RSP)

	MOVD nN_OCSTART(RSP), R0
	MOVD R0, nN_OC(RSP)

neon1d_loop_oc:
	MOVD nN_OC(RSP), R0
	MOVD nN_OCSTART(RSP), R1
	MOVD nN_OCPERG(RSP), R2
	ADD  R2, R1, R1
	CMP  R1, R0
	BGE  neon1d_next_g

	// wRowP = weight + oc*icPerGroup*kSize*8
	MOVD nN_OC(RSP), R0
	MOVD nN_ICPERG(RSP), R1
	MUL  R1, R0, R0
	MOVD nN_KSIZE(RSP), R1
	MUL  R1, R0, R0
	LSL  $3, R0, R0
	MOVD nN_WP(RSP), R1
	ADD  R1, R0, R0
	MOVD R0, nN_WROWP(RSP)

	MOVD $0, R0
	MOVD R0, nN_LO(RSP)

neon1d_loop_lo:
	MOVD nN_LO(RSP), R0
	MOVD nN_LOUT(RSP), R1
	CMP  R1, R0
	BGE  neon1d_next_oc

	// sum = bias[oc]
	MOVD nN_BP(RSP), R0
	MOVD nN_OC(RSP), R1
	LSL  $3, R1, R1
	ADD  R1, R0, R0
	FMOVD (R0), F0
	FMOVD F0, nN_SUM(RSP)

	MOVD $0, R0
	MOVD R0, nN_IC(RSP)

neon1d_loop_ic:
	MOVD nN_IC(RSP), R0
	MOVD nN_ICPERG(RSP), R1
	CMP  R1, R0
	BGE  neon1d_write_out

	// absIC = icStart + ic
	MOVD nN_ICSTART(RSP), R0
	MOVD nN_IC(RSP), R1
	ADD  R1, R0, R0
	// inputBase = (ni*inC + absIC) * l
	MOVD nN_NI(RSP), R1
	MOVD nN_INC(RSP), R2
	MUL  R2, R1, R1
	ADD  R0, R1, R1
	MOVD nN_L(RSP), R2
	MUL  R2, R1, R1        // R1 = inputBase

	// weightBase = ic * kSize
	MOVD nN_IC(RSP), R2
	MOVD nN_KSIZE(RSP), R3
	MUL  R3, R2, R2        // R2 = weightBase (element index)

	// Check fast path: dilation==1 and all in-bounds
	MOVD nN_DIL(RSP), R3
	CMP  $1, R3
	BNE  neon1d_scalar_k

	MOVD nN_LO(RSP), R3
	MOVD nN_STRIDE(RSP), R4
	MUL  R4, R3, R3        // R3 = lo*stride
	MOVD nN_PAD(RSP), R4
	CMP  R4, R3
	BLT  neon1d_scalar_k

	MOVD R3, R5
	SUB  R4, R5, R5        // R5 = lo*stride - pad
	MOVD nN_KSIZE(RSP), R4
	ADD  R4, R5, R5
	SUB  $1, R5, R5        // R5 = lo*stride - pad + kSize - 1
	MOVD nN_L(RSP), R4
	CMP  R4, R5
	BGE  neon1d_scalar_k

	// Fast NEON path
	MOVD nN_PAD(RSP), R4
	SUB  R4, R3, R3        // R3 = inputIdx = lo*stride - pad

	// xPtr = xP + (inputBase + R3)*8
	MOVD R1, R4
	ADD  R3, R4, R4
	LSL  $3, R4, R4
	MOVD nN_XP(RSP), R5
	ADD  R5, R4, R4        // R4 = xPtr

	// wPtr = wRowP + weightBase*8
	LSL  $3, R2, R5
	MOVD nN_WROWP(RSP), R6
	ADD  R6, R5, R5        // R5 = wPtr

	MOVD nN_KSIZE(RSP), R6
	FMOVD nN_SUM(RSP), F0
	FMOVD $0.0, F9         // accumulator

neon1d_fast_k:
	CMP  $4, R6
	BLT  neon1d_fast_tail
	FMOVD.P 8(R4), F1
	FMOVD.P 8(R5), F2
	FMADDD F1, F2, F9, F9
	FMOVD.P 8(R4), F3
	FMOVD.P 8(R5), F4
	FMADDD F3, F4, F9, F9
	FMOVD.P 8(R4), F5
	FMOVD.P 8(R5), F6
	FMADDD F5, F6, F9, F9
	FMOVD.P 8(R4), F7
	FMOVD.P 8(R5), F8
	FMADDD F7, F8, F9, F9
	SUB  $4, R6, R6
	B    neon1d_fast_k
neon1d_fast_tail:
	FADDD F9, F0, F0
	CBZ  R6, neon1d_fast_done
neon1d_fast_scalar:
	FMOVD.P 8(R4), F1
	FMOVD.P 8(R5), F2
	FMADDD F1, F2, F0, F0
	SUBS $1, R6, R6
	BNE  neon1d_fast_scalar
neon1d_fast_done:
	FMOVD F0, nN_SUM(RSP)
	JMP  neon1d_next_ic

neon1d_scalar_k:
	MOVD nN_KSIZE(RSP), R3
	MOVD $0, R4
	FMOVD nN_SUM(RSP), F0
neon1d_sk:
	CMP  R3, R4
	BGE  neon1d_sk_done

	// inputIdx = lo*stride + k*dilation - pad
	MOVD nN_LO(RSP), R5
	MOVD nN_STRIDE(RSP), R6
	MUL  R6, R5, R5
	MOVD nN_DIL(RSP), R6
	MUL  R6, R4, R6
	ADD  R6, R5, R5
	MOVD nN_PAD(RSP), R6
	SUB  R6, R5, R5        // R5 = inputIdx

	// bounds check
	CMP  $0, R5; BLT neon1d_sk_next  // negative
	MOVD nN_L(RSP), R6
	CMP  R6, R5
	BGE  neon1d_sk_next

	// x[inputBase + inputIdx]
	MOVD R1, R6
	ADD  R5, R6, R6
	LSL  $3, R6, R6
	MOVD nN_XP(RSP), R7
	FMOVD (R7)(R6), F1

	// weight[weightBase + k]
	MOVD R2, R6
	ADD  R4, R6, R6
	LSL  $3, R6, R6
	MOVD nN_WROWP(RSP), R7
	FMOVD (R7)(R6), F2
	FMADDD F1, F2, F0, F0
neon1d_sk_next:
	ADD  $1, R4, R4
	B    neon1d_sk
neon1d_sk_done:
	FMOVD F0, nN_SUM(RSP)

neon1d_next_ic:
	MOVD nN_IC(RSP), R0
	ADD  $1, R0, R0
	MOVD R0, nN_IC(RSP)
	B    neon1d_loop_ic

neon1d_write_out:
	// out[ni*outC*lOut + oc*lOut + lo] = sum
	MOVD nN_NI(RSP), R0
	MOVD nN_OUTC(RSP), R1
	MUL  R1, R0, R0
	MOVD nN_LOUT(RSP), R1
	MUL  R1, R0, R0
	MOVD nN_OC(RSP), R1
	MOVD nN_LOUT(RSP), R2
	MUL  R2, R1, R1
	ADD  R1, R0, R0
	MOVD nN_LO(RSP), R1
	ADD  R1, R0, R0
	LSL  $3, R0, R0
	MOVD nN_OUTP(RSP), R1
	FMOVD nN_SUM(RSP), F0
	FMOVD F0, (R1)(R0)

	MOVD nN_LO(RSP), R0
	ADD  $1, R0, R0
	MOVD R0, nN_LO(RSP)
	B    neon1d_loop_lo

neon1d_next_oc:
	MOVD nN_OC(RSP), R0
	ADD  $1, R0, R0
	MOVD R0, nN_OC(RSP)
	B    neon1d_loop_oc

neon1d_next_g:
	MOVD nN_GRP(RSP), R0
	ADD  $1, R0, R0
	MOVD R0, nN_GRP(RSP)
	B    neon1d_loop_g

neon1d_next_ni:
	MOVD nN_NI(RSP), R0
	ADD  $1, R0, R0
	MOVD R0, nN_NI(RSP)
	B    neon1d_loop_ni

neon1d_done:
	RET
