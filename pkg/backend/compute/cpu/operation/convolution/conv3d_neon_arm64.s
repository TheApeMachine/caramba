#include "textflag.h"

// conv3dNEON — full conv3d forward pass using ARM NEON/FP.
//
// func conv3dNEON(out, x, weight, bias []float64,
//                 n, inC, d, h, w, outC,
//                 kD, kH, kW, sD, sH, sW, pD, pH, pW,
//                 dilD, dilH, dilW, groups, dOut, hOut, wOut int)
//
// Arg offsets from FP:
//   out:0 x:24 weight:48 bias:72
//   n:96 inC:104 d:112 h:120 w:128 outC:136
//   kD:144 kH:152 kW:160
//   sD:168 sH:176 sW:184
//   pD:192 pH:200 pW:208
//   dilD:216 dilH:224 dilW:232
//   groups:240 dOut:248 hOut:256 wOut:264
//   Total: 272 bytes
//
// Frame: 640 bytes
#define n3OUTP    0
#define n3XP      8
#define n3WP      16
#define n3BP      24
#define n3N       32
#define n3INC     40
#define n3D       48
#define n3H       56
#define n3W       64
#define n3OUTC    72
#define n3KD      80
#define n3KH      88
#define n3KW      96
#define n3SD      104
#define n3SH      112
#define n3SW      120
#define n3PD      128
#define n3PH      136
#define n3PW      144
#define n3DD      152
#define n3DH      160
#define n3DW      168
#define n3GROUPS  176
#define n3DOUT    184
#define n3HOUT    192
#define n3WOUT    200
#define n3NI      208
#define n3GRP     216
#define n3OC      224
#define n3DO      232
#define n3HO      240
#define n3WO      248
#define n3IC      256
#define n3KD_IDX  264
#define n3KH_IDX  272
#define n3ICPERG  280
#define n3OCPERG  288
#define n3OCSTART 296
#define n3ICSTART 304
#define n3WROWP   312
#define n3SUM     320
#define n3INBASE  328
#define n3WBASE   336
#define n3DI      344
#define n3HI      352

TEXT ·conv3dNEON(SB), $640-272
	MOVD out+0(FP), R0;     MOVD R0, n3OUTP(RSP)
	MOVD x+24(FP), R0;      MOVD R0, n3XP(RSP)
	MOVD weight+48(FP), R0; MOVD R0, n3WP(RSP)
	MOVD bias+72(FP), R0;   MOVD R0, n3BP(RSP)
	MOVD n+96(FP), R0;      MOVD R0, n3N(RSP)
	MOVD inC+104(FP), R0;   MOVD R0, n3INC(RSP)
	MOVD d+112(FP), R0;     MOVD R0, n3D(RSP)
	MOVD h+120(FP), R0;     MOVD R0, n3H(RSP)
	MOVD w+128(FP), R0;     MOVD R0, n3W(RSP)
	MOVD outC+136(FP), R0;  MOVD R0, n3OUTC(RSP)
	MOVD kD+144(FP), R0;    MOVD R0, n3KD(RSP)
	MOVD kH+152(FP), R0;    MOVD R0, n3KH(RSP)
	MOVD kW+160(FP), R0;    MOVD R0, n3KW(RSP)
	MOVD sD+168(FP), R0;    MOVD R0, n3SD(RSP)
	MOVD sH+176(FP), R0;    MOVD R0, n3SH(RSP)
	MOVD sW+184(FP), R0;    MOVD R0, n3SW(RSP)
	MOVD pD+192(FP), R0;    MOVD R0, n3PD(RSP)
	MOVD pH+200(FP), R0;    MOVD R0, n3PH(RSP)
	MOVD pW+208(FP), R0;    MOVD R0, n3PW(RSP)
	MOVD dilD+216(FP), R0;  MOVD R0, n3DD(RSP)
	MOVD dilH+224(FP), R0;  MOVD R0, n3DH(RSP)
	MOVD dilW+232(FP), R0;  MOVD R0, n3DW(RSP)
	MOVD groups+240(FP), R0; MOVD R0, n3GROUPS(RSP)
	MOVD dOut+248(FP), R0;  MOVD R0, n3DOUT(RSP)
	MOVD hOut+256(FP), R0;  MOVD R0, n3HOUT(RSP)
	MOVD wOut+264(FP), R0;  MOVD R0, n3WOUT(RSP)

	MOVD $0, R0; MOVD R0, n3NI(RSP)

n3_loop_ni:
	MOVD n3NI(RSP), R0; MOVD n3N(RSP), R1; CMP R1, R0; BGE n3_done

	MOVD n3INC(RSP), R0; MOVD n3GROUPS(RSP), R1; UDIV R1, R0, R2; MOVD R2, n3ICPERG(RSP)
	MOVD n3OUTC(RSP), R0; MOVD n3GROUPS(RSP), R1; UDIV R1, R0, R2; MOVD R2, n3OCPERG(RSP)

	MOVD $0, R0; MOVD R0, n3GRP(RSP)

n3_loop_g:
	MOVD n3GRP(RSP), R0; MOVD n3GROUPS(RSP), R1; CMP R1, R0; BGE n3_next_ni

	MOVD n3GRP(RSP), R0; MOVD n3OCPERG(RSP), R1; MUL R1, R0, R2; MOVD R2, n3OCSTART(RSP)
	MOVD n3GRP(RSP), R0; MOVD n3ICPERG(RSP), R1; MUL R1, R0, R2; MOVD R2, n3ICSTART(RSP)

	MOVD n3OCSTART(RSP), R0; MOVD R0, n3OC(RSP)

n3_loop_oc:
	MOVD n3OC(RSP), R0
	MOVD n3OCSTART(RSP), R1; MOVD n3OCPERG(RSP), R2; ADD R2, R1, R1; CMP R1, R0; BGE n3_next_g

	MOVD n3OC(RSP), R0; MOVD n3ICPERG(RSP), R1; MUL R1, R0, R0
	MOVD n3KD(RSP), R1; MUL R1, R0, R0; MOVD n3KH(RSP), R1; MUL R1, R0, R0; MOVD n3KW(RSP), R1; MUL R1, R0, R0
	LSL $3, R0, R0; MOVD n3WP(RSP), R1; ADD R1, R0, R0; MOVD R0, n3WROWP(RSP)

	MOVD $0, R0; MOVD R0, n3DO(RSP)

n3_loop_do:
	MOVD n3DO(RSP), R0; MOVD n3DOUT(RSP), R1; CMP R1, R0; BGE n3_next_oc

	MOVD $0, R0; MOVD R0, n3HO(RSP)

n3_loop_ho:
	MOVD n3HO(RSP), R0; MOVD n3HOUT(RSP), R1; CMP R1, R0; BGE n3_next_do

	MOVD $0, R0; MOVD R0, n3WO(RSP)

n3_loop_wo:
	MOVD n3WO(RSP), R0; MOVD n3WOUT(RSP), R1; CMP R1, R0; BGE n3_next_ho

	MOVD n3BP(RSP), R0; MOVD n3OC(RSP), R1; LSL $3, R1, R1; ADD R1, R0, R0
	FMOVD (R0), F0; FMOVD F0, n3SUM(RSP)

	MOVD $0, R0; MOVD R0, n3IC(RSP)

n3_loop_ic:
	MOVD n3IC(RSP), R0; MOVD n3ICPERG(RSP), R1; CMP R1, R0; BGE n3_write_out

	MOVD n3ICSTART(RSP), R0; MOVD n3IC(RSP), R1; ADD R1, R0, R0
	MOVD n3NI(RSP), R1; MOVD n3INC(RSP), R2; MUL R2, R1, R1; ADD R0, R1, R1
	MOVD n3D(RSP), R2; MUL R2, R1, R1; MOVD n3H(RSP), R2; MUL R2, R1, R1; MOVD n3W(RSP), R2; MUL R2, R1, R1
	MOVD R1, n3INBASE(RSP)

	MOVD n3IC(RSP), R1; MOVD n3KD(RSP), R2; MUL R2, R1, R1; MOVD n3KH(RSP), R2; MUL R2, R1, R1; MOVD n3KW(RSP), R2; MUL R2, R1, R1
	MOVD R1, n3WBASE(RSP)

	MOVD $0, R0; MOVD R0, n3KD_IDX(RSP)

n3_loop_kd:
	MOVD n3KD_IDX(RSP), R0; MOVD n3KD(RSP), R1; CMP R1, R0; BGE n3_next_ic

	MOVD n3DO(RSP), R1; MOVD n3SD(RSP), R2; MUL R2, R1, R1
	MOVD n3KD_IDX(RSP), R2; MOVD n3DD(RSP), R3; MUL R3, R2, R2; ADD R2, R1, R1
	MOVD n3PD(RSP), R2; SUB R2, R1, R1
	MOVD R1, n3DI(RSP)
	CMP $0, R1; BLT n3_next_kd
	MOVD n3D(RSP), R2; CMP R2, R1; BGE n3_next_kd

	MOVD $0, R0; MOVD R0, n3KH_IDX(RSP)

n3_loop_kh:
	MOVD n3KH_IDX(RSP), R0; MOVD n3KH(RSP), R1; CMP R1, R0; BGE n3_next_kd

	MOVD n3HO(RSP), R1; MOVD n3SH(RSP), R2; MUL R2, R1, R1
	MOVD n3KH_IDX(RSP), R2; MOVD n3DH(RSP), R3; MUL R3, R2, R2; ADD R2, R1, R1
	MOVD n3PH(RSP), R2; SUB R2, R1, R1
	MOVD R1, n3HI(RSP)
	CMP $0, R1; BLT n3_next_kh
	MOVD n3H(RSP), R2; CMP R2, R1; BGE n3_next_kh

	MOVD n3DW(RSP), R2; CMP $1, R2; BNE n3_kw_scatter

	MOVD n3WO(RSP), R2; MOVD n3SW(RSP), R3; MUL R3, R2, R2
	MOVD n3PW(RSP), R3; CMP R3, R2; BLT n3_kw_scatter

	MOVD R2, R4; SUB R3, R4, R4; MOVD n3KW(RSP), R3; ADD R3, R4, R4; SUB $1, R4, R4
	MOVD n3W(RSP), R3; CMP R3, R4; BGE n3_kw_scatter

	MOVD n3PW(RSP), R3; SUB R3, R2, R2  // wi_start

	MOVD n3INBASE(RSP), R3
	MOVD n3DI(RSP), R4; MOVD n3H(RSP), R5; MUL R5, R4, R4; MOVD n3W(RSP), R5; MUL R5, R4, R4; ADD R4, R3, R3
	MOVD n3HI(RSP), R4; MOVD n3W(RSP), R5; MUL R5, R4, R4; ADD R4, R3, R3
	ADD R2, R3, R3; LSL $3, R3, R3; MOVD n3XP(RSP), R4; ADD R4, R3, R3

	MOVD n3WBASE(RSP), R4
	MOVD n3KD_IDX(RSP), R5; MOVD n3KH(RSP), R6; MUL R6, R5, R5; MOVD n3KW(RSP), R6; MUL R6, R5, R5; ADD R5, R4, R4
	MOVD n3KH_IDX(RSP), R5; MOVD n3KW(RSP), R6; MUL R6, R5, R5; ADD R5, R4, R4
	LSL $3, R4, R4; MOVD n3WROWP(RSP), R5; ADD R5, R4, R4

	MOVD n3KW(RSP), R5
	FMOVD n3SUM(RSP), F0; FMOVD $0.0, F9

n3_neon_kw:
	CMP $4, R5; BLT n3_neon_kw_tail
	FMOVD.P 8(R3), F1; FMOVD.P 8(R4), F2; FMADDD F1, F2, F9, F9
	FMOVD.P 8(R3), F3; FMOVD.P 8(R4), F4; FMADDD F3, F4, F9, F9
	FMOVD.P 8(R3), F5; FMOVD.P 8(R4), F6; FMADDD F5, F6, F9, F9
	FMOVD.P 8(R3), F7; FMOVD.P 8(R4), F8; FMADDD F7, F8, F9, F9
	SUB $4, R5, R5; B n3_neon_kw
n3_neon_kw_tail:
	FADDD F9, F0, F0
	CBZ R5, n3_neon_kw_done
n3_neon_kw_sc:
	FMOVD.P 8(R3), F1; FMOVD.P 8(R4), F2; FMADDD F1, F2, F0, F0
	SUBS $1, R5, R5; BNE n3_neon_kw_sc
n3_neon_kw_done:
	FMOVD F0, n3SUM(RSP); B n3_next_kh

n3_kw_scatter:
	MOVD $0, R2; FMOVD n3SUM(RSP), F0
n3_kw_sc_loop:
	MOVD n3KW(RSP), R3; CMP R3, R2; BGE n3_kw_sc_done

	MOVD n3WO(RSP), R3; MOVD n3SW(RSP), R4; MUL R4, R3, R3
	MOVD R2, R4; MOVD n3DW(RSP), R5; MUL R5, R4, R4; ADD R4, R3, R3
	MOVD n3PW(RSP), R4; SUB R4, R3, R3  // wi

	CMP $0, R3; BLT n3_kw_sc_next
	MOVD n3W(RSP), R4; CMP R4, R3; BGE n3_kw_sc_next

	MOVD n3INBASE(RSP), R4
	MOVD n3DI(RSP), R5; MOVD n3H(RSP), R6; MUL R6, R5, R5; MOVD n3W(RSP), R6; MUL R6, R5, R5; ADD R5, R4, R4
	MOVD n3HI(RSP), R5; MOVD n3W(RSP), R6; MUL R6, R5, R5; ADD R5, R4, R4; ADD R3, R4, R4
	LSL $3, R4, R4; MOVD n3XP(RSP), R5; FMOVD (R5)(R4), F1

	MOVD n3WBASE(RSP), R4
	MOVD n3KD_IDX(RSP), R5; MOVD n3KH(RSP), R6; MUL R6, R5, R5; MOVD n3KW(RSP), R6; MUL R6, R5, R5; ADD R5, R4, R4
	MOVD n3KH_IDX(RSP), R5; MOVD n3KW(RSP), R6; MUL R6, R5, R5; ADD R5, R4, R4; ADD R2, R4, R4
	LSL $3, R4, R4; MOVD n3WROWP(RSP), R5; FMOVD (R5)(R4), F2
	FMADDD F1, F2, F0, F0
n3_kw_sc_next:
	ADD $1, R2, R2; B n3_kw_sc_loop
n3_kw_sc_done:
	FMOVD F0, n3SUM(RSP)

n3_next_kh:
	MOVD n3KH_IDX(RSP), R0; ADD $1, R0, R0; MOVD R0, n3KH_IDX(RSP); B n3_loop_kh

n3_next_kd:
	MOVD n3KD_IDX(RSP), R0; ADD $1, R0, R0; MOVD R0, n3KD_IDX(RSP); B n3_loop_kd

n3_next_ic:
	MOVD n3IC(RSP), R0; ADD $1, R0, R0; MOVD R0, n3IC(RSP); B n3_loop_ic

n3_write_out:
	MOVD n3NI(RSP), R0; MOVD n3OUTC(RSP), R1; MUL R1, R0, R0
	MOVD n3DOUT(RSP), R1; MUL R1, R0, R0; MOVD n3HOUT(RSP), R1; MUL R1, R0, R0; MOVD n3WOUT(RSP), R1; MUL R1, R0, R0
	MOVD n3OC(RSP), R1
	MOVD n3DOUT(RSP), R2; MUL R2, R1, R1; MOVD n3HOUT(RSP), R2; MUL R2, R1, R1; MOVD n3WOUT(RSP), R2; MUL R2, R1, R1; ADD R1, R0, R0
	MOVD n3DO(RSP), R1; MOVD n3HOUT(RSP), R2; MUL R2, R1, R1; MOVD n3WOUT(RSP), R2; MUL R2, R1, R1; ADD R1, R0, R0
	MOVD n3HO(RSP), R1; MOVD n3WOUT(RSP), R2; MUL R2, R1, R1; ADD R1, R0, R0
	MOVD n3WO(RSP), R1; ADD R1, R0, R0
	LSL $3, R0, R0; MOVD n3OUTP(RSP), R1
	FMOVD n3SUM(RSP), F0; FMOVD F0, (R1)(R0)

	MOVD n3WO(RSP), R0; ADD $1, R0, R0; MOVD R0, n3WO(RSP); B n3_loop_wo

n3_next_ho:
	MOVD n3HO(RSP), R0; ADD $1, R0, R0; MOVD R0, n3HO(RSP); B n3_loop_ho

n3_next_do:
	MOVD n3DO(RSP), R0; ADD $1, R0, R0; MOVD R0, n3DO(RSP); B n3_loop_do

n3_next_oc:
	MOVD n3OC(RSP), R0; ADD $1, R0, R0; MOVD R0, n3OC(RSP); B n3_loop_oc

n3_next_g:
	MOVD n3GRP(RSP), R0; ADD $1, R0, R0; MOVD R0, n3GRP(RSP); B n3_loop_g

n3_next_ni:
	MOVD n3NI(RSP), R0; ADD $1, R0, R0; MOVD R0, n3NI(RSP); B n3_loop_ni

n3_done:
	RET
