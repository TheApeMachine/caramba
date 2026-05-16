#include "textflag.h"

// conv3dSSE2 — full conv3d forward pass using SSE2.
//
// func conv3dSSE2(out, x, weight, bias []float64,
//                 n, inC, d, h, w, outC,
//                 kD, kH, kW, sD, sH, sW, pD, pH, pW,
//                 dilD, dilH, dilW, groups, dOut, hOut, wOut int)
//
// Arg offsets from FP:
//   out:0  x:24  weight:48  bias:72
//   n:96  inC:104  d:112  h:120  w:128  outC:136
//   kD:144 kH:152 kW:160
//   sD:168 sH:176 sW:184
//   pD:192 pH:200 pW:208
//   dilD:216 dilH:224 dilW:232
//   groups:240 dOut:248 hOut:256 wOut:264
//   Total args: 272 bytes
//
// Frame: 640 bytes
#define c3OUTP    0
#define c3XP      8
#define c3WP      16
#define c3BP      24
#define c3N       32
#define c3INC     40
#define c3D       48
#define c3H       56
#define c3W       64
#define c3OUTC    72
#define c3KD      80
#define c3KH      88
#define c3KW      96
#define c3SD      104
#define c3SH      112
#define c3SW      120
#define c3PD      128
#define c3PH      136
#define c3PW      144
#define c3DD      152
#define c3DH      160
#define c3DW      168
#define c3GROUPS  176
#define c3DOUT    184
#define c3HOUT    192
#define c3WOUT    200
#define c3NI      208
#define c3GRP     216
#define c3OC      224
#define c3DO      232
#define c3HO      240
#define c3WO      248
#define c3IC      256
#define c3KD_IDX  264
#define c3KH_IDX  272
#define c3ICPERG  280
#define c3OCPERG  288
#define c3OCSTART 296
#define c3ICSTART 304
#define c3WROWP   312
#define c3SUM     320
#define c3INBASE  328
#define c3WBASE   336
#define c3DI      344
#define c3HI      352

TEXT ·conv3dSSE2(SB), $640-272
	MOVQ out+0(FP), AX;     MOVQ AX, c3OUTP(SP)
	MOVQ x+24(FP), AX;      MOVQ AX, c3XP(SP)
	MOVQ weight+48(FP), AX; MOVQ AX, c3WP(SP)
	MOVQ bias+72(FP), AX;   MOVQ AX, c3BP(SP)
	MOVQ n+96(FP), AX;      MOVQ AX, c3N(SP)
	MOVQ inC+104(FP), AX;   MOVQ AX, c3INC(SP)
	MOVQ d+112(FP), AX;     MOVQ AX, c3D(SP)
	MOVQ h+120(FP), AX;     MOVQ AX, c3H(SP)
	MOVQ w+128(FP), AX;     MOVQ AX, c3W(SP)
	MOVQ outC+136(FP), AX;  MOVQ AX, c3OUTC(SP)
	MOVQ kD+144(FP), AX;    MOVQ AX, c3KD(SP)
	MOVQ kH+152(FP), AX;    MOVQ AX, c3KH(SP)
	MOVQ kW+160(FP), AX;    MOVQ AX, c3KW(SP)
	MOVQ sD+168(FP), AX;    MOVQ AX, c3SD(SP)
	MOVQ sH+176(FP), AX;    MOVQ AX, c3SH(SP)
	MOVQ sW+184(FP), AX;    MOVQ AX, c3SW(SP)
	MOVQ pD+192(FP), AX;    MOVQ AX, c3PD(SP)
	MOVQ pH+200(FP), AX;    MOVQ AX, c3PH(SP)
	MOVQ pW+208(FP), AX;    MOVQ AX, c3PW(SP)
	MOVQ dilD+216(FP), AX;  MOVQ AX, c3DD(SP)
	MOVQ dilH+224(FP), AX;  MOVQ AX, c3DH(SP)
	MOVQ dilW+232(FP), AX;  MOVQ AX, c3DW(SP)
	MOVQ groups+240(FP), AX; MOVQ AX, c3GROUPS(SP)
	MOVQ dOut+248(FP), AX;  MOVQ AX, c3DOUT(SP)
	MOVQ hOut+256(FP), AX;  MOVQ AX, c3HOUT(SP)
	MOVQ wOut+264(FP), AX;  MOVQ AX, c3WOUT(SP)

	XORQ AX, AX; MOVQ AX, c3NI(SP)

ss3_loop_ni:
	MOVQ c3NI(SP), AX; MOVQ c3N(SP), BX; CMPQ AX, BX; JGE ss3_done

	MOVQ c3INC(SP), AX; MOVQ c3GROUPS(SP), BX; XORQ DX, DX; DIVQ BX; MOVQ AX, c3ICPERG(SP)
	MOVQ c3OUTC(SP), AX; MOVQ c3GROUPS(SP), BX; XORQ DX, DX; DIVQ BX; MOVQ AX, c3OCPERG(SP)

	XORQ AX, AX; MOVQ AX, c3GRP(SP)

ss3_loop_g:
	MOVQ c3GRP(SP), AX; MOVQ c3GROUPS(SP), BX; CMPQ AX, BX; JGE ss3_next_ni

	MOVQ c3GRP(SP), AX; MOVQ c3OCPERG(SP), BX; IMULQ BX, AX; MOVQ AX, c3OCSTART(SP)
	MOVQ c3GRP(SP), AX; MOVQ c3ICPERG(SP), BX; IMULQ BX, AX; MOVQ AX, c3ICSTART(SP)

	MOVQ c3OCSTART(SP), AX; MOVQ AX, c3OC(SP)

ss3_loop_oc:
	MOVQ c3OC(SP), AX
	MOVQ c3OCSTART(SP), BX; MOVQ c3OCPERG(SP), CX; ADDQ BX, CX; CMPQ AX, CX; JGE ss3_next_g

	MOVQ c3OC(SP), AX; MOVQ c3ICPERG(SP), BX; IMULQ BX, AX
	MOVQ c3KD(SP), BX; IMULQ BX, AX; MOVQ c3KH(SP), BX; IMULQ BX, AX; MOVQ c3KW(SP), BX; IMULQ BX, AX
	SHLQ $3, AX; MOVQ c3WP(SP), BX; ADDQ BX, AX; MOVQ AX, c3WROWP(SP)

	XORQ AX, AX; MOVQ AX, c3DO(SP)

ss3_loop_do:
	MOVQ c3DO(SP), AX; MOVQ c3DOUT(SP), BX; CMPQ AX, BX; JGE ss3_next_oc

	XORQ AX, AX; MOVQ AX, c3HO(SP)

ss3_loop_ho:
	MOVQ c3HO(SP), AX; MOVQ c3HOUT(SP), BX; CMPQ AX, BX; JGE ss3_next_do

	XORQ AX, AX; MOVQ AX, c3WO(SP)

ss3_loop_wo:
	MOVQ c3WO(SP), AX; MOVQ c3WOUT(SP), BX; CMPQ AX, BX; JGE ss3_next_ho

	MOVQ c3BP(SP), AX; MOVQ c3OC(SP), BX; SHLQ $3, BX; MOVSD (AX)(BX*1), X15; MOVSD X15, c3SUM(SP)

	XORQ AX, AX; MOVQ AX, c3IC(SP)

ss3_loop_ic:
	MOVQ c3IC(SP), AX; MOVQ c3ICPERG(SP), BX; CMPQ AX, BX; JGE ss3_write_out

	MOVQ c3ICSTART(SP), AX; MOVQ c3IC(SP), BX; ADDQ BX, AX
	MOVQ c3NI(SP), BX; MOVQ c3INC(SP), CX; IMULQ CX, BX; ADDQ AX, BX
	MOVQ c3D(SP), CX; IMULQ CX, BX; MOVQ c3H(SP), CX; IMULQ CX, BX; MOVQ c3W(SP), CX; IMULQ CX, BX
	MOVQ BX, c3INBASE(SP)

	MOVQ c3IC(SP), BX; MOVQ c3KD(SP), CX; IMULQ CX, BX; MOVQ c3KH(SP), CX; IMULQ CX, BX; MOVQ c3KW(SP), CX; IMULQ CX, BX
	MOVQ BX, c3WBASE(SP)

	XORQ AX, AX; MOVQ AX, c3KD_IDX(SP)

ss3_loop_kd:
	MOVQ c3KD_IDX(SP), AX; MOVQ c3KD(SP), BX; CMPQ AX, BX; JGE ss3_next_ic

	MOVQ c3DO(SP), BX; MOVQ c3SD(SP), CX; IMULQ CX, BX
	MOVQ c3KD_IDX(SP), CX; MOVQ c3DD(SP), DX; IMULQ DX, CX; ADDQ CX, BX
	MOVQ c3PD(SP), CX; SUBQ CX, BX
	MOVQ BX, c3DI(SP)
	CMPQ BX, $0; JL ss3_next_kd
	MOVQ c3D(SP), CX; CMPQ BX, CX; JGE ss3_next_kd

	XORQ AX, AX; MOVQ AX, c3KH_IDX(SP)

ss3_loop_kh:
	MOVQ c3KH_IDX(SP), AX; MOVQ c3KH(SP), BX; CMPQ AX, BX; JGE ss3_next_kd

	MOVQ c3HO(SP), BX; MOVQ c3SH(SP), CX; IMULQ CX, BX
	MOVQ c3KH_IDX(SP), CX; MOVQ c3DH(SP), DX; IMULQ DX, CX; ADDQ CX, BX
	MOVQ c3PH(SP), CX; SUBQ CX, BX
	MOVQ BX, c3HI(SP)
	CMPQ BX, $0; JL ss3_next_kh
	MOVQ c3H(SP), CX; CMPQ BX, CX; JGE ss3_next_kh

	MOVQ c3DW(SP), R8; CMPQ R8, $1; JNE ss3_kw_scatter

	MOVQ c3WO(SP), R8; MOVQ c3SW(SP), R9; IMULQ R9, R8
	MOVQ c3PW(SP), R9; CMPQ R8, R9; JL ss3_kw_scatter

	MOVQ R8, R10; SUBQ R9, R10; MOVQ c3KW(SP), R11; ADDQ R11, R10; SUBQ $1, R10
	MOVQ c3W(SP), R11; CMPQ R10, R11; JGE ss3_kw_scatter

	MOVQ c3WO(SP), R10; MOVQ c3SW(SP), R11; IMULQ R11, R10; MOVQ c3PW(SP), R11; SUBQ R11, R10

	MOVQ c3INBASE(SP), R11
	MOVQ c3DI(SP), R12; MOVQ c3H(SP), R13; IMULQ R13, R12; MOVQ c3W(SP), R13; IMULQ R13, R12; ADDQ R12, R11
	MOVQ c3HI(SP), R12; MOVQ c3W(SP), R13; IMULQ R13, R12; ADDQ R12, R11
	ADDQ R10, R11; SHLQ $3, R11; MOVQ c3XP(SP), R12; ADDQ R12, R11

	MOVQ c3WBASE(SP), R12
	MOVQ c3KD_IDX(SP), R13; MOVQ c3KH(SP), R14; IMULQ R14, R13; MOVQ c3KW(SP), R14; IMULQ R14, R13; ADDQ R13, R12
	MOVQ c3KH_IDX(SP), R13; MOVQ c3KW(SP), R14; IMULQ R14, R13; ADDQ R13, R12
	SHLQ $3, R12; MOVQ c3WROWP(SP), R13; ADDQ R13, R12

	MOVQ c3KW(SP), R13
	MOVSD c3SUM(SP), X0; XORPS X8, X8

ss3_sse2_kw:
	CMPQ R13, $2; JL ss3_sse2_kw_tail
	MOVUPD 0(R11), X1; MOVUPD 0(R12), X2; MULPD X2, X1; ADDPD X1, X8
	ADDQ $16, R11; ADDQ $16, R12; SUBQ $2, R13; JMP ss3_sse2_kw
ss3_sse2_kw_tail:
	HADDPD X8, X8; ADDSD X8, X0
	CMPQ R13, $0; JLE ss3_sse2_kw_done
ss3_sse2_kw_sc:
	MOVSD (R11), X1; MOVSD (R12), X2; MULSD X2, X1; ADDSD X1, X0
	ADDQ $8, R11; ADDQ $8, R12; DECQ R13; JNZ ss3_sse2_kw_sc
ss3_sse2_kw_done:
	MOVSD X0, c3SUM(SP); JMP ss3_next_kh

ss3_kw_scatter:
	XORQ R8, R8; MOVSD c3SUM(SP), X0
ss3_kw_sc_loop:
	MOVQ c3KW(SP), R9; CMPQ R8, R9; JGE ss3_kw_sc_done

	MOVQ c3WO(SP), R9; MOVQ c3SW(SP), R10; IMULQ R10, R9
	MOVQ R8, R10; MOVQ c3DW(SP), R11; IMULQ R11, R10; ADDQ R10, R9
	MOVQ c3PW(SP), R10; SUBQ R10, R9

	CMPQ R9, $0; JL ss3_kw_sc_next
	MOVQ c3W(SP), R10; CMPQ R9, R10; JGE ss3_kw_sc_next

	MOVQ c3INBASE(SP), R10
	MOVQ c3DI(SP), R11; MOVQ c3H(SP), R12; IMULQ R12, R11; MOVQ c3W(SP), R12; IMULQ R12, R11; ADDQ R11, R10
	MOVQ c3HI(SP), R11; MOVQ c3W(SP), R12; IMULQ R12, R11; ADDQ R11, R10; ADDQ R9, R10
	SHLQ $3, R10; MOVQ c3XP(SP), R11; MOVSD (R11)(R10*1), X1

	MOVQ c3WBASE(SP), R10
	MOVQ c3KD_IDX(SP), R11; MOVQ c3KH(SP), R12; IMULQ R12, R11; MOVQ c3KW(SP), R12; IMULQ R12, R11; ADDQ R11, R10
	MOVQ c3KH_IDX(SP), R11; MOVQ c3KW(SP), R12; IMULQ R12, R11; ADDQ R11, R10; ADDQ R8, R10
	SHLQ $3, R10; MOVQ c3WROWP(SP), R11; MOVSD (R11)(R10*1), X2
	MULSD X2, X1; ADDSD X1, X0
ss3_kw_sc_next:
	INCQ R8; JMP ss3_kw_sc_loop
ss3_kw_sc_done:
	MOVSD X0, c3SUM(SP)

ss3_next_kh:
	MOVQ c3KH_IDX(SP), AX; INCQ AX; MOVQ AX, c3KH_IDX(SP); JMP ss3_loop_kh

ss3_next_kd:
	MOVQ c3KD_IDX(SP), AX; INCQ AX; MOVQ AX, c3KD_IDX(SP); JMP ss3_loop_kd

ss3_next_ic:
	MOVQ c3IC(SP), AX; INCQ AX; MOVQ AX, c3IC(SP); JMP ss3_loop_ic

ss3_write_out:
	MOVQ c3NI(SP), AX; MOVQ c3OUTC(SP), BX; IMULQ BX, AX
	MOVQ c3DOUT(SP), BX; IMULQ BX, AX; MOVQ c3HOUT(SP), BX; IMULQ BX, AX; MOVQ c3WOUT(SP), BX; IMULQ BX, AX
	MOVQ c3OC(SP), BX
	MOVQ c3DOUT(SP), CX; IMULQ CX, BX; MOVQ c3HOUT(SP), CX; IMULQ CX, BX; MOVQ c3WOUT(SP), CX; IMULQ CX, BX; ADDQ BX, AX
	MOVQ c3DO(SP), BX; MOVQ c3HOUT(SP), CX; IMULQ CX, BX; MOVQ c3WOUT(SP), CX; IMULQ CX, BX; ADDQ BX, AX
	MOVQ c3HO(SP), BX; MOVQ c3WOUT(SP), CX; IMULQ CX, BX; ADDQ BX, AX
	ADDQ c3WO(SP), AX; SHLQ $3, AX
	MOVQ c3OUTP(SP), BX; MOVSD c3SUM(SP), X0; MOVSD X0, (BX)(AX*1)

	MOVQ c3WO(SP), AX; INCQ AX; MOVQ AX, c3WO(SP); JMP ss3_loop_wo

ss3_next_ho:
	MOVQ c3HO(SP), AX; INCQ AX; MOVQ AX, c3HO(SP); JMP ss3_loop_ho

ss3_next_do:
	MOVQ c3DO(SP), AX; INCQ AX; MOVQ AX, c3DO(SP); JMP ss3_loop_do

ss3_next_oc:
	MOVQ c3OC(SP), AX; INCQ AX; MOVQ AX, c3OC(SP); JMP ss3_loop_oc

ss3_next_g:
	MOVQ c3GRP(SP), AX; INCQ AX; MOVQ AX, c3GRP(SP); JMP ss3_loop_g

ss3_next_ni:
	MOVQ c3NI(SP), AX; INCQ AX; MOVQ AX, c3NI(SP); JMP ss3_loop_ni

ss3_done:
	RET
