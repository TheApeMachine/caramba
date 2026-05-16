#include "textflag.h"

// conv2dSSE2 — full conv2d forward pass using SSE2.
//
// func conv2dSSE2(out, x, weight, bias []float64,
//                 n, inC, h, w, outC, kH, kW,
//                 strideH, strideW, padH, padW, dilH, dilW,
//                 groups, hOut, wOut int)
//
// Arg offsets from FP:
//   out:     0   x:    24   weight: 48   bias: 72
//   n:      96   inC: 104   h:     112   w:   120
//   outC:  128   kH:  136   kW:    144
//   strideH:152  strideW:160 padH:168   padW:176
//   dilH:  184   dilW: 192  groups:200  hOut:208  wOut:216
//   Total args: 224 bytes
//
// Frame slots (SP-relative, frame=448):
#define c2OUTP    0
#define c2XP      8
#define c2WP      16
#define c2BP      24
#define c2N       32
#define c2INC     40
#define c2H       48
#define c2W       56
#define c2OUTC    64
#define c2KH      72
#define c2KW      80
#define c2SH      88
#define c2SW      96
#define c2PH      104
#define c2PW      112
#define c2DH      120
#define c2DW      128
#define c2GROUPS  136
#define c2HOUT    144
#define c2WOUT    152
#define c2NI      160
#define c2GRP     168
#define c2OC      176
#define c2HO      184
#define c2WO      192
#define c2IC      200
#define c2KH_IDX  208  // kh loop var
#define c2ICPERG  216
#define c2OCPERG  224
#define c2OCSTART 232
#define c2ICSTART 240
#define c2WROWP   248
#define c2SUM     256

TEXT ·conv2dSSE2(SB), $448-224
	MOVQ out+0(FP), AX;     MOVQ AX, c2OUTP(SP)
	MOVQ x+24(FP), AX;      MOVQ AX, c2XP(SP)
	MOVQ weight+48(FP), AX; MOVQ AX, c2WP(SP)
	MOVQ bias+72(FP), AX;   MOVQ AX, c2BP(SP)
	MOVQ n+96(FP), AX;      MOVQ AX, c2N(SP)
	MOVQ inC+104(FP), AX;   MOVQ AX, c2INC(SP)
	MOVQ h+112(FP), AX;     MOVQ AX, c2H(SP)
	MOVQ w+120(FP), AX;     MOVQ AX, c2W(SP)
	MOVQ outC+128(FP), AX;  MOVQ AX, c2OUTC(SP)
	MOVQ kH+136(FP), AX;    MOVQ AX, c2KH(SP)
	MOVQ kW+144(FP), AX;    MOVQ AX, c2KW(SP)
	MOVQ strideH+152(FP), AX; MOVQ AX, c2SH(SP)
	MOVQ strideW+160(FP), AX; MOVQ AX, c2SW(SP)
	MOVQ padH+168(FP), AX;  MOVQ AX, c2PH(SP)
	MOVQ padW+176(FP), AX;  MOVQ AX, c2PW(SP)
	MOVQ dilH+184(FP), AX;  MOVQ AX, c2DH(SP)
	MOVQ dilW+192(FP), AX;  MOVQ AX, c2DW(SP)
	MOVQ groups+200(FP), AX; MOVQ AX, c2GROUPS(SP)
	MOVQ hOut+208(FP), AX;  MOVQ AX, c2HOUT(SP)
	MOVQ wOut+216(FP), AX;  MOVQ AX, c2WOUT(SP)

	XORQ AX, AX; MOVQ AX, c2NI(SP)

s2_loop_ni:
	MOVQ c2NI(SP), AX; MOVQ c2N(SP), BX; CMPQ AX, BX; JGE s2_done

	MOVQ c2INC(SP), AX; MOVQ c2GROUPS(SP), BX; XORQ DX, DX; DIVQ BX; MOVQ AX, c2ICPERG(SP)
	MOVQ c2OUTC(SP), AX; MOVQ c2GROUPS(SP), BX; XORQ DX, DX; DIVQ BX; MOVQ AX, c2OCPERG(SP)

	XORQ AX, AX; MOVQ AX, c2GRP(SP)

s2_loop_g:
	MOVQ c2GRP(SP), AX; MOVQ c2GROUPS(SP), BX; CMPQ AX, BX; JGE s2_next_ni

	MOVQ c2GRP(SP), AX; MOVQ c2OCPERG(SP), BX; IMULQ BX, AX; MOVQ AX, c2OCSTART(SP)
	MOVQ c2GRP(SP), AX; MOVQ c2ICPERG(SP), BX; IMULQ BX, AX; MOVQ AX, c2ICSTART(SP)

	MOVQ c2OCSTART(SP), AX; MOVQ AX, c2OC(SP)

s2_loop_oc:
	MOVQ c2OC(SP), AX
	MOVQ c2OCSTART(SP), BX; MOVQ c2OCPERG(SP), CX; ADDQ BX, CX
	CMPQ AX, CX; JGE s2_next_g

	MOVQ c2OC(SP), AX
	MOVQ c2ICPERG(SP), BX; IMULQ BX, AX
	MOVQ c2KH(SP), BX; IMULQ BX, AX
	MOVQ c2KW(SP), BX; IMULQ BX, AX
	SHLQ $3, AX; MOVQ c2WP(SP), BX; ADDQ BX, AX; MOVQ AX, c2WROWP(SP)

	XORQ AX, AX; MOVQ AX, c2HO(SP)

s2_loop_ho:
	MOVQ c2HO(SP), AX; MOVQ c2HOUT(SP), BX; CMPQ AX, BX; JGE s2_next_oc

	XORQ AX, AX; MOVQ AX, c2WO(SP)

s2_loop_wo:
	MOVQ c2WO(SP), AX; MOVQ c2WOUT(SP), BX; CMPQ AX, BX; JGE s2_next_ho

	MOVQ c2BP(SP), AX; MOVQ c2OC(SP), BX; SHLQ $3, BX; MOVSD (AX)(BX*1), X15; MOVSD X15, c2SUM(SP)

	XORQ AX, AX; MOVQ AX, c2IC(SP)

s2_loop_ic:
	MOVQ c2IC(SP), AX; MOVQ c2ICPERG(SP), BX; CMPQ AX, BX; JGE s2_write_out

	MOVQ c2ICSTART(SP), AX; MOVQ c2IC(SP), BX; ADDQ BX, AX
	MOVQ c2NI(SP), BX; MOVQ c2INC(SP), CX; IMULQ CX, BX; ADDQ AX, BX
	MOVQ c2H(SP), CX; IMULQ CX, BX; MOVQ c2W(SP), CX; IMULQ CX, BX
	MOVQ c2IC(SP), DX; MOVQ c2KH(SP), CX; IMULQ CX, DX; MOVQ c2KW(SP), CX; IMULQ CX, DX

	XORQ R8, R8; MOVQ R8, c2KH_IDX(SP)

s2_loop_kh:
	MOVQ c2KH_IDX(SP), R8; MOVQ c2KH(SP), R9; CMPQ R8, R9; JGE s2_next_ic

	MOVQ c2HO(SP), R9; MOVQ c2SH(SP), R10; IMULQ R10, R9
	MOVQ R8, R10; MOVQ c2DH(SP), R11; IMULQ R11, R10; ADDQ R10, R9
	MOVQ c2PH(SP), R10; SUBQ R10, R9

	CMPQ R9, $0; JL s2_next_kh
	MOVQ c2H(SP), R10; CMPQ R9, R10; JGE s2_next_kh

	MOVQ c2DW(SP), R10; CMPQ R10, $1; JNE s2_kw_scatter

	MOVQ c2WO(SP), R10; MOVQ c2SW(SP), R11; IMULQ R11, R10
	MOVQ c2PW(SP), R11; CMPQ R10, R11; JL s2_kw_scatter

	MOVQ R10, R12; SUBQ R11, R12
	MOVQ c2KW(SP), R11; ADDQ R11, R12; SUBQ $1, R12
	MOVQ c2W(SP), R11; CMPQ R12, R11; JGE s2_kw_scatter

	// Fast SSE2 path
	MOVQ c2WO(SP), R12; MOVQ c2SW(SP), R13; IMULQ R13, R12
	MOVQ c2PW(SP), R13; SUBQ R13, R12

	MOVQ BX, R13; MOVQ R9, R14; MOVQ c2W(SP), R15; IMULQ R15, R14; ADDQ R14, R13; ADDQ R12, R13
	SHLQ $3, R13; MOVQ c2XP(SP), R14; ADDQ R14, R13

	MOVQ DX, R14; MOVQ c2KH_IDX(SP), R15; MOVQ c2KW(SP), R10; IMULQ R10, R15; ADDQ R15, R14
	SHLQ $3, R14; MOVQ c2WROWP(SP), R15; ADDQ R15, R14

	MOVQ c2KW(SP), R15
	MOVSD c2SUM(SP), X0; XORPS X8, X8

s2_sse2_kw:
	CMPQ R15, $2; JL s2_sse2_kw_tail
	MOVUPD 0(R13), X1; MOVUPD 0(R14), X2; MULPD X2, X1; ADDPD X1, X8
	ADDQ $16, R13; ADDQ $16, R14; SUBQ $2, R15; JMP s2_sse2_kw
s2_sse2_kw_tail:
	MOVAPD X8, X9; UNPCKHPD X8, X9; ADDSD X9, X8; ADDSD X8, X0
	CMPQ R15, $0; JLE s2_sse2_kw_done
s2_sse2_kw_sc:
	MOVSD (R13), X1; MOVSD (R14), X2; MULSD X2, X1; ADDSD X1, X0
	ADDQ $8, R13; ADDQ $8, R14; DECQ R15; JNZ s2_sse2_kw_sc
s2_sse2_kw_done:
	MOVSD X0, c2SUM(SP); JMP s2_next_kh

s2_kw_scatter:
	XORQ R10, R10; MOVSD c2SUM(SP), X0
s2_kw_sc_loop:
	MOVQ c2KW(SP), R11; CMPQ R10, R11; JGE s2_kw_sc_done

	MOVQ c2WO(SP), R11; MOVQ c2SW(SP), R12; IMULQ R12, R11
	MOVQ R10, R12; MOVQ c2DW(SP), R13; IMULQ R13, R12; ADDQ R12, R11
	MOVQ c2PW(SP), R12; SUBQ R12, R11

	CMPQ R11, $0; JL s2_kw_sc_next
	MOVQ c2W(SP), R12; CMPQ R11, R12; JGE s2_kw_sc_next

	MOVQ BX, R12; MOVQ R9, R13; MOVQ c2W(SP), R14; IMULQ R14, R13; ADDQ R13, R12; ADDQ R11, R12
	SHLQ $3, R12; MOVQ c2XP(SP), R13; MOVSD (R13)(R12*1), X1

	MOVQ DX, R12; MOVQ c2KH_IDX(SP), R13; MOVQ c2KW(SP), R14; IMULQ R14, R13; ADDQ R13, R12; ADDQ R10, R12
	SHLQ $3, R12; MOVQ c2WROWP(SP), R13; MOVSD (R13)(R12*1), X2
	MULSD X2, X1; ADDSD X1, X0
s2_kw_sc_next:
	INCQ R10; JMP s2_kw_sc_loop
s2_kw_sc_done:
	MOVSD X0, c2SUM(SP)

s2_next_kh:
	MOVQ c2KH_IDX(SP), R8; INCQ R8; MOVQ R8, c2KH_IDX(SP); JMP s2_loop_kh

s2_next_ic:
	MOVQ c2IC(SP), AX; INCQ AX; MOVQ AX, c2IC(SP); JMP s2_loop_ic

s2_write_out:
	MOVQ c2NI(SP), AX; MOVQ c2OUTC(SP), BX; IMULQ BX, AX
	MOVQ c2HOUT(SP), BX; IMULQ BX, AX; MOVQ c2WOUT(SP), BX; IMULQ BX, AX
	MOVQ c2OC(SP), BX; MOVQ c2HOUT(SP), CX; IMULQ CX, BX; MOVQ c2WOUT(SP), CX; IMULQ CX, BX; ADDQ BX, AX
	MOVQ c2HO(SP), BX; MOVQ c2WOUT(SP), CX; IMULQ CX, BX; ADDQ BX, AX
	ADDQ c2WO(SP), AX; SHLQ $3, AX
	MOVQ c2OUTP(SP), BX; MOVSD c2SUM(SP), X0; MOVSD X0, (BX)(AX*1)

	MOVQ c2WO(SP), AX; INCQ AX; MOVQ AX, c2WO(SP); JMP s2_loop_wo

s2_next_ho:
	MOVQ c2HO(SP), AX; INCQ AX; MOVQ AX, c2HO(SP); JMP s2_loop_ho

s2_next_oc:
	MOVQ c2OC(SP), AX; INCQ AX; MOVQ AX, c2OC(SP); JMP s2_loop_oc

s2_next_g:
	MOVQ c2GRP(SP), AX; INCQ AX; MOVQ AX, c2GRP(SP); JMP s2_loop_g

s2_next_ni:
	MOVQ c2NI(SP), AX; INCQ AX; MOVQ AX, c2NI(SP); JMP s2_loop_ni

s2_done:
	RET
