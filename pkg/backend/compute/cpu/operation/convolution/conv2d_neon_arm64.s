#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// conv2dNEON — full conv2d forward pass using ARM NEON/FP.
//
// func conv2dNEON(out, x, weight, bias []float64,
//                 n, inC, h, w, outC, kH, kW,
//                 strideH, strideW, padH, padW, dilH, dilW,
//                 groups, hOut, wOut int)
//
// Arg offsets from FP (each slice 24 bytes, each int 8 bytes):
//   out:0 x:24 weight:48 bias:72
//   n:96 inC:104 h:112 w:120 outC:128
//   kH:136 kW:144 strideH:152 strideW:160
//   padH:168 padW:176 dilH:184 dilW:192
//   groups:200 hOut:208 wOut:216
//   Total: 224 bytes
//
// Frame: 448 bytes
#define n2OUTP    0
#define n2XP      8
#define n2WP      16
#define n2BP      24
#define n2N       32
#define n2INC     40
#define n2H       48
#define n2W       56
#define n2OUTC    64
#define n2KH      72
#define n2KW      80
#define n2SH      88
#define n2SW      96
#define n2PH      104
#define n2PW      112
#define n2DH      120
#define n2DW      128
#define n2GROUPS  136
#define n2HOUT    144
#define n2WOUT    152
#define n2NI      160
#define n2GRP     168
#define n2OC      176
#define n2HO      184
#define n2WO      192
#define n2IC      200
#define n2KH_IDX  208
#define n2ICPERG  216
#define n2OCPERG  224
#define n2OCSTART 232
#define n2ICSTART 240
#define n2WROWP   248
#define n2SUM     256
#define n2INPUTBASE 264
#define n2WBASE   272
#define n2HI      280

TEXT ·conv2dNEON(SB), $448-224
	MOVD out+0(FP), R0;     MOVD R0, n2OUTP(RSP)
	MOVD x+24(FP), R0;      MOVD R0, n2XP(RSP)
	MOVD weight+48(FP), R0; MOVD R0, n2WP(RSP)
	MOVD bias+72(FP), R0;   MOVD R0, n2BP(RSP)
	MOVD n+96(FP), R0;      MOVD R0, n2N(RSP)
	MOVD inC+104(FP), R0;   MOVD R0, n2INC(RSP)
	MOVD h+112(FP), R0;     MOVD R0, n2H(RSP)
	MOVD w+120(FP), R0;     MOVD R0, n2W(RSP)
	MOVD outC+128(FP), R0;  MOVD R0, n2OUTC(RSP)
	MOVD kH+136(FP), R0;    MOVD R0, n2KH(RSP)
	MOVD kW+144(FP), R0;    MOVD R0, n2KW(RSP)
	MOVD strideH+152(FP), R0; MOVD R0, n2SH(RSP)
	MOVD strideW+160(FP), R0; MOVD R0, n2SW(RSP)
	MOVD padH+168(FP), R0;  MOVD R0, n2PH(RSP)
	MOVD padW+176(FP), R0;  MOVD R0, n2PW(RSP)
	MOVD dilH+184(FP), R0;  MOVD R0, n2DH(RSP)
	MOVD dilW+192(FP), R0;  MOVD R0, n2DW(RSP)
	MOVD groups+200(FP), R0; MOVD R0, n2GROUPS(RSP)
	MOVD hOut+208(FP), R0;  MOVD R0, n2HOUT(RSP)
	MOVD wOut+216(FP), R0;  MOVD R0, n2WOUT(RSP)

	MOVD $0, R0; MOVD R0, n2NI(RSP)

n2_loop_ni:
	MOVD n2NI(RSP), R0; MOVD n2N(RSP), R1; CMP R1, R0; BGE n2_done

	MOVD n2INC(RSP), R0; MOVD n2GROUPS(RSP), R1; UDIV R1, R0, R2; MOVD R2, n2ICPERG(RSP)
	MOVD n2OUTC(RSP), R0; MOVD n2GROUPS(RSP), R1; UDIV R1, R0, R2; MOVD R2, n2OCPERG(RSP)

	MOVD $0, R0; MOVD R0, n2GRP(RSP)

n2_loop_g:
	MOVD n2GRP(RSP), R0; MOVD n2GROUPS(RSP), R1; CMP R1, R0; BGE n2_next_ni

	MOVD n2GRP(RSP), R0; MOVD n2OCPERG(RSP), R1; MUL R1, R0, R2; MOVD R2, n2OCSTART(RSP)
	MOVD n2GRP(RSP), R0; MOVD n2ICPERG(RSP), R1; MUL R1, R0, R2; MOVD R2, n2ICSTART(RSP)

	MOVD n2OCSTART(RSP), R0; MOVD R0, n2OC(RSP)

n2_loop_oc:
	MOVD n2OC(RSP), R0
	MOVD n2OCSTART(RSP), R1; MOVD n2OCPERG(RSP), R2; ADD R2, R1, R1
	CMP R1, R0; BGE n2_next_g

	// wRowP = weight + oc*icPerGroup*kH*kW*8
	MOVD n2OC(RSP), R0; MOVD n2ICPERG(RSP), R1; MUL R1, R0, R0
	MOVD n2KH(RSP), R1; MUL R1, R0, R0
	MOVD n2KW(RSP), R1; MUL R1, R0, R0
	LSL $3, R0, R0; MOVD n2WP(RSP), R1; ADD R1, R0, R0; MOVD R0, n2WROWP(RSP)

	MOVD $0, R0; MOVD R0, n2HO(RSP)

n2_loop_ho:
	MOVD n2HO(RSP), R0; MOVD n2HOUT(RSP), R1; CMP R1, R0; BGE n2_next_oc

	MOVD $0, R0; MOVD R0, n2WO(RSP)

n2_loop_wo:
	MOVD n2WO(RSP), R0; MOVD n2WOUT(RSP), R1; CMP R1, R0; BGE n2_next_ho

	MOVD n2BP(RSP), R0; MOVD n2OC(RSP), R1; LSL $3, R1, R1; ADD R1, R0, R0
	FMOVD (R0), F0; FMOVD F0, n2SUM(RSP)

	MOVD $0, R0; MOVD R0, n2IC(RSP)

n2_loop_ic:
	MOVD n2IC(RSP), R0; MOVD n2ICPERG(RSP), R1; CMP R1, R0; BGE n2_write_out

	// absIC = icStart + ic; inputBase = (ni*inC + absIC)*h*w
	MOVD n2ICSTART(RSP), R0; MOVD n2IC(RSP), R1; ADD R1, R0, R0
	MOVD n2NI(RSP), R1; MOVD n2INC(RSP), R2; MUL R2, R1, R1; ADD R0, R1, R1
	MOVD n2H(RSP), R2; MUL R2, R1, R1; MOVD n2W(RSP), R2; MUL R2, R1, R1
	MOVD R1, n2INPUTBASE(RSP)

	// weightBase = ic * kH * kW
	MOVD n2IC(RSP), R1; MOVD n2KH(RSP), R2; MUL R2, R1, R1; MOVD n2KW(RSP), R2; MUL R2, R1, R1
	MOVD R1, n2WBASE(RSP)

	MOVD $0, R0; MOVD R0, n2KH_IDX(RSP)

n2_loop_kh:
	MOVD n2KH_IDX(RSP), R0; MOVD n2KH(RSP), R1; CMP R1, R0; BGE n2_next_ic

	// hi = ho*sH + kh*dilH - padH
	MOVD n2HO(RSP), R1; MOVD n2SH(RSP), R2; MUL R2, R1, R1
	MOVD n2KH_IDX(RSP), R2; MOVD n2DH(RSP), R3; MUL R3, R2, R2; ADD R2, R1, R1
	MOVD n2PH(RSP), R2; SUB R2, R1, R1  // R1 = hi
	MOVD R1, n2HI(RSP)

	CMP $0, R1; BLT n2_next_kh  // hi < 0
	MOVD n2H(RSP), R2; CMP R2, R1; BGE n2_next_kh

	// Check fast path: dilW==1, wo*sW >= padW, wo*sW+(kW-1) < w+padW
	MOVD n2DW(RSP), R2; CMP $1, R2; BNE n2_kw_scatter

	MOVD n2WO(RSP), R2; MOVD n2SW(RSP), R3; MUL R3, R2, R2
	MOVD n2PW(RSP), R3; CMP R3, R2; BLT n2_kw_scatter

	MOVD R2, R4; SUB R3, R4, R4
	MOVD n2KW(RSP), R3; ADD R3, R4, R4; SUB $1, R4, R4
	MOVD n2W(RSP), R3; CMP R3, R4; BGE n2_kw_scatter

	// Fast NEON path for kW row
	MOVD n2PW(RSP), R3; SUB R3, R2, R2  // R2 = wo*sW - padW = wi_start

	// xPtr = xP + (inputBase + hi*w + wi_start)*8
	MOVD n2INPUTBASE(RSP), R3
	MOVD n2HI(RSP), R4; MOVD n2W(RSP), R5; MUL R5, R4, R4; ADD R4, R3, R3; ADD R2, R3, R3
	LSL $3, R3, R3; MOVD n2XP(RSP), R4; ADD R4, R3, R3

	// wPtr = wRowP + (weightBase + kh_idx*kW)*8
	MOVD n2WBASE(RSP), R4
	MOVD n2KH_IDX(RSP), R5; MOVD n2KW(RSP), R6; MUL R6, R5, R5; ADD R5, R4, R4
	LSL $3, R4, R4; MOVD n2WROWP(RSP), R5; ADD R5, R4, R4

	MOVD n2KW(RSP), R5
	FMOVD n2SUM(RSP), F0; VEOR V9.B16, V9.B16, V9.B16

n2_neon_kw:
	CMP $2, R5; BLT n2_neon_kw_tail
	VLD1.P 16(R3), [V1.D2]
	VLD1.P 16(R4), [V2.D2]
	VFMUL_D2(2, 1, 3)
	VFADD_D2(3, 9, 9)
	SUB $2, R5, R5; B n2_neon_kw
n2_neon_kw_tail:
	ADD $400, RSP, R6
	VST1.P [V9.D2], 16(R6)
	FMOVD 400(RSP), F9
	FMOVD 408(RSP), F10
	FADDD F10, F9, F9
	FADDD F9, F0, F0
	CBZ R5, n2_neon_kw_done
n2_neon_kw_sc:
	FMOVD.P 8(R3), F1; FMOVD.P 8(R4), F2; FMADDD F1, F0, F2, F0
	SUBS $1, R5, R5; BNE n2_neon_kw_sc
n2_neon_kw_done:
	FMOVD F0, n2SUM(RSP); B n2_next_kh

n2_kw_scatter:
	MOVD $0, R2; FMOVD n2SUM(RSP), F0
n2_kw_sc_loop:
	MOVD n2KW(RSP), R3; CMP R3, R2; BGE n2_kw_sc_done

	MOVD n2WO(RSP), R3; MOVD n2SW(RSP), R4; MUL R4, R3, R3
	MOVD R2, R4; MOVD n2DW(RSP), R5; MUL R5, R4, R4; ADD R4, R3, R3
	MOVD n2PW(RSP), R4; SUB R4, R3, R3  // R3 = wi

	CMP $0, R3; BLT n2_kw_sc_next
	MOVD n2W(RSP), R4; CMP R4, R3; BGE n2_kw_sc_next

	// x[inputBase + hi*w + wi]
	MOVD n2INPUTBASE(RSP), R4
	MOVD n2HI(RSP), R5; MOVD n2W(RSP), R6; MUL R6, R5, R5; ADD R5, R4, R4; ADD R3, R4, R4
	LSL $3, R4, R4; MOVD n2XP(RSP), R5; FMOVD (R5)(R4), F1

	// weight[weightBase + kh*kW + kw]
	MOVD n2WBASE(RSP), R4
	MOVD n2KH_IDX(RSP), R5; MOVD n2KW(RSP), R6; MUL R6, R5, R5; ADD R5, R4, R4; ADD R2, R4, R4
	LSL $3, R4, R4; MOVD n2WROWP(RSP), R5; FMOVD (R5)(R4), F2
	FMADDD F1, F0, F2, F0
n2_kw_sc_next:
	ADD $1, R2, R2; B n2_kw_sc_loop
n2_kw_sc_done:
	FMOVD F0, n2SUM(RSP)

n2_next_kh:
	MOVD n2KH_IDX(RSP), R0; ADD $1, R0, R0; MOVD R0, n2KH_IDX(RSP); B n2_loop_kh

n2_next_ic:
	MOVD n2IC(RSP), R0; ADD $1, R0, R0; MOVD R0, n2IC(RSP); B n2_loop_ic

n2_write_out:
	MOVD n2NI(RSP), R0; MOVD n2OUTC(RSP), R1; MUL R1, R0, R0
	MOVD n2HOUT(RSP), R1; MUL R1, R0, R0; MOVD n2WOUT(RSP), R1; MUL R1, R0, R0
	MOVD n2OC(RSP), R1; MOVD n2HOUT(RSP), R2; MUL R2, R1, R1; MOVD n2WOUT(RSP), R2; MUL R2, R1, R1; ADD R1, R0, R0
	MOVD n2HO(RSP), R1; MOVD n2WOUT(RSP), R2; MUL R2, R1, R1; ADD R1, R0, R0
	MOVD n2WO(RSP), R1; ADD R1, R0, R0
	LSL $3, R0, R0; MOVD n2OUTP(RSP), R1
	FMOVD n2SUM(RSP), F0; FMOVD F0, (R1)(R0)

	MOVD n2WO(RSP), R0; ADD $1, R0, R0; MOVD R0, n2WO(RSP); B n2_loop_wo

n2_next_ho:
	MOVD n2HO(RSP), R0; ADD $1, R0, R0; MOVD R0, n2HO(RSP); B n2_loop_ho

n2_next_oc:
	MOVD n2OC(RSP), R0; ADD $1, R0, R0; MOVD R0, n2OC(RSP); B n2_loop_oc

n2_next_g:
	MOVD n2GRP(RSP), R0; ADD $1, R0, R0; MOVD R0, n2GRP(RSP); B n2_loop_g

n2_next_ni:
	MOVD n2NI(RSP), R0; ADD $1, R0, R0; MOVD R0, n2NI(RSP); B n2_loop_ni

n2_done:
	RET
