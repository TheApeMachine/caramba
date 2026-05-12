#include "textflag.h"

// conv2dAVX2 — full conv2d forward pass using AVX2+FMA.
//
// func conv2dAVX2(out, x, weight, bias []float64,
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

TEXT ·conv2dAVX2(SB), $448-224
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

c2_loop_ni:
	MOVQ c2NI(SP), AX; MOVQ c2N(SP), BX; CMPQ AX, BX; JGE c2_done

	MOVQ c2INC(SP), AX; MOVQ c2GROUPS(SP), BX; XORQ DX, DX; DIVQ BX; MOVQ AX, c2ICPERG(SP)
	MOVQ c2OUTC(SP), AX; MOVQ c2GROUPS(SP), BX; XORQ DX, DX; DIVQ BX; MOVQ AX, c2OCPERG(SP)

	XORQ AX, AX; MOVQ AX, c2GRP(SP)

c2_loop_g:
	MOVQ c2GRP(SP), AX; MOVQ c2GROUPS(SP), BX; CMPQ AX, BX; JGE c2_next_ni

	MOVQ c2GRP(SP), AX; MOVQ c2OCPERG(SP), BX; IMULQ BX, AX; MOVQ AX, c2OCSTART(SP)
	MOVQ c2GRP(SP), AX; MOVQ c2ICPERG(SP), BX; IMULQ BX, AX; MOVQ AX, c2ICSTART(SP)

	MOVQ c2OCSTART(SP), AX; MOVQ AX, c2OC(SP)

c2_loop_oc:
	MOVQ c2OC(SP), AX
	MOVQ c2OCSTART(SP), BX; MOVQ c2OCPERG(SP), CX; ADDQ BX, CX
	CMPQ AX, CX; JGE c2_next_g

	// wRowP = weight + oc * icPerGroup * kH * kW * 8
	MOVQ c2OC(SP), AX
	MOVQ c2ICPERG(SP), BX; IMULQ BX, AX
	MOVQ c2KH(SP), BX; IMULQ BX, AX
	MOVQ c2KW(SP), BX; IMULQ BX, AX
	SHLQ $3, AX
	MOVQ c2WP(SP), BX; ADDQ BX, AX; MOVQ AX, c2WROWP(SP)

	XORQ AX, AX; MOVQ AX, c2HO(SP)

c2_loop_ho:
	MOVQ c2HO(SP), AX; MOVQ c2HOUT(SP), BX; CMPQ AX, BX; JGE c2_next_oc

	XORQ AX, AX; MOVQ AX, c2WO(SP)

c2_loop_wo:
	MOVQ c2WO(SP), AX; MOVQ c2WOUT(SP), BX; CMPQ AX, BX; JGE c2_next_ho

	// sum = bias[oc]
	MOVQ c2BP(SP), AX; MOVQ c2OC(SP), BX; SHLQ $3, BX; MOVSD (AX)(BX*1), X15; MOVSD X15, c2SUM(SP)

	XORQ AX, AX; MOVQ AX, c2IC(SP)

c2_loop_ic:
	MOVQ c2IC(SP), AX; MOVQ c2ICPERG(SP), BX; CMPQ AX, BX; JGE c2_write_out

	// absIC = icStart + ic
	MOVQ c2ICSTART(SP), AX; MOVQ c2IC(SP), BX; ADDQ BX, AX
	// inputBase = (ni*inC + absIC) * h * w
	MOVQ c2NI(SP), BX; MOVQ c2INC(SP), CX; IMULQ CX, BX; ADDQ AX, BX
	MOVQ c2H(SP), CX; IMULQ CX, BX
	MOVQ c2W(SP), CX; IMULQ CX, BX  // BX = inputBase (element index)

	// weightBase = ic * kH * kW
	MOVQ c2IC(SP), DX; MOVQ c2KH(SP), CX; IMULQ CX, DX; MOVQ c2KW(SP), CX; IMULQ CX, DX

	// kh loop
	XORQ R8, R8; MOVQ R8, c2KH_IDX(SP)

c2_loop_kh:
	MOVQ c2KH_IDX(SP), R8; MOVQ c2KH(SP), R9; CMPQ R8, R9; JGE c2_next_ic

	// hi = ho*strideH + kh*dilH - padH
	MOVQ c2HO(SP), R9; MOVQ c2SH(SP), R10; IMULQ R10, R9
	MOVQ R8, R10; MOVQ c2DH(SP), R11; IMULQ R11, R10; ADDQ R10, R9
	MOVQ c2PH(SP), R10; SUBQ R10, R9  // R9 = hi

	// bounds check hi
	CMPQ R9, $0; JL c2_next_kh
	MOVQ c2H(SP), R10; CMPQ R9, R10; JGE c2_next_kh

	// Check if full kW row is in-bounds (dilW==1, wo*strideW >= padW, wo*strideW+(kW-1) < w+padW)
	MOVQ c2DW(SP), R10; CMPQ R10, $1; JNE c2_kw_scatter

	MOVQ c2WO(SP), R10; MOVQ c2SW(SP), R11; IMULQ R11, R10
	MOVQ c2PW(SP), R11; CMPQ R10, R11; JL c2_kw_scatter

	MOVQ R10, R12; SUBQ R11, R12
	MOVQ c2KW(SP), R11; ADDQ R11, R12; SUBQ $1, R12
	MOVQ c2W(SP), R11; CMPQ R12, R11; JGE c2_kw_scatter

	// Fast path: contiguous kW row
	// inputIdx = wo*strideW - padW (start of kW row)
	MOVQ c2WO(SP), R12; MOVQ c2SW(SP), R13; IMULQ R13, R12
	MOVQ c2PW(SP), R13; SUBQ R13, R12

	// xPtr = xP + (inputBase + hi*w + inputIdx)*8
	MOVQ BX, R13; MOVQ R9, R14; MOVQ c2W(SP), R15; IMULQ R15, R14; ADDQ R14, R13; ADDQ R12, R13
	SHLQ $3, R13; MOVQ c2XP(SP), R14; ADDQ R14, R13

	// wPtr = wRowP + (weightBase + kh*kW)*8
	MOVQ DX, R14; MOVQ c2KH_IDX(SP), R15; MOVQ c2KW(SP), R10; IMULQ R10, R15; ADDQ R15, R14
	SHLQ $3, R14; MOVQ c2WROWP(SP), R15; ADDQ R15, R14

	MOVQ c2KW(SP), R15
	MOVSD c2SUM(SP), X0
	VXORPD Y1, Y1, Y1

c2_avx2_kw:
	CMPQ R15, $4; JL c2_avx2_kw_tail
	VMOVUPD  0(R13), Y2; VMOVUPD  0(R14), Y3
	VFMADD231PD Y2, Y3, Y1
	ADDQ $32, R13; ADDQ $32, R14; SUBQ $4, R15; JMP c2_avx2_kw
c2_avx2_kw_tail:
	VEXTRACTF128 $1, Y1, X2; VADDPD X2, X1, X1; VHADDPD X1, X1, X1; ADDSD X1, X0
	CMPQ R15, $0; JLE c2_avx2_kw_done
c2_avx2_kw_sc:
	MOVSD (R13), X1; MOVSD (R14), X2; MULSD X2, X1; ADDSD X1, X0
	ADDQ $8, R13; ADDQ $8, R14; DECQ R15; JNZ c2_avx2_kw_sc
c2_avx2_kw_done:
	MOVSD X0, c2SUM(SP)
	VZEROUPPER
	JMP c2_next_kh

c2_kw_scatter:
	// Scalar kW loop with boundary per element
	XORQ R10, R10  // kw = 0
	MOVSD c2SUM(SP), X0
c2_kw_sc_loop:
	MOVQ c2KW(SP), R11; CMPQ R10, R11; JGE c2_kw_sc_done

	// wi = wo*strideW + kw*dilW - padW
	MOVQ c2WO(SP), R11; MOVQ c2SW(SP), R12; IMULQ R12, R11
	MOVQ R10, R12; MOVQ c2DW(SP), R13; IMULQ R13, R12; ADDQ R12, R11
	MOVQ c2PW(SP), R12; SUBQ R12, R11  // R11 = wi

	CMPQ R11, $0; JL c2_kw_sc_next
	MOVQ c2W(SP), R12; CMPQ R11, R12; JGE c2_kw_sc_next

	// x[inputBase + hi*w + wi]
	MOVQ BX, R12; MOVQ R9, R13; MOVQ c2W(SP), R14; IMULQ R14, R13; ADDQ R13, R12; ADDQ R11, R12
	SHLQ $3, R12; MOVQ c2XP(SP), R13; MOVSD (R13)(R12*1), X1

	// weight[weightBase + kh*kW + kw]
	MOVQ DX, R12; MOVQ c2KH_IDX(SP), R13; MOVQ c2KW(SP), R14; IMULQ R14, R13; ADDQ R13, R12; ADDQ R10, R12
	SHLQ $3, R12; MOVQ c2WROWP(SP), R13; MOVSD (R13)(R12*1), X2
	MULSD X2, X1; ADDSD X1, X0
c2_kw_sc_next:
	INCQ R10; JMP c2_kw_sc_loop
c2_kw_sc_done:
	MOVSD X0, c2SUM(SP)

c2_next_kh:
	MOVQ c2KH_IDX(SP), R8; INCQ R8; MOVQ R8, c2KH_IDX(SP); JMP c2_loop_kh

c2_next_ic:
	MOVQ c2IC(SP), AX; INCQ AX; MOVQ AX, c2IC(SP); JMP c2_loop_ic

c2_write_out:
	// out[ni*outC*hOut*wOut + oc*hOut*wOut + ho*wOut + wo]
	MOVQ c2NI(SP), AX; MOVQ c2OUTC(SP), BX; IMULQ BX, AX
	MOVQ c2HOUT(SP), BX; IMULQ BX, AX; MOVQ c2WOUT(SP), BX; IMULQ BX, AX
	MOVQ c2OC(SP), BX; MOVQ c2HOUT(SP), CX; IMULQ CX, BX; MOVQ c2WOUT(SP), CX; IMULQ CX, BX; ADDQ BX, AX
	MOVQ c2HO(SP), BX; MOVQ c2WOUT(SP), CX; IMULQ CX, BX; ADDQ BX, AX
	ADDQ c2WO(SP), AX; SHLQ $3, AX
	MOVQ c2OUTP(SP), BX; MOVSD c2SUM(SP), X0; MOVSD X0, (BX)(AX*1)

	MOVQ c2WO(SP), AX; INCQ AX; MOVQ AX, c2WO(SP); JMP c2_loop_wo

c2_next_ho:
	MOVQ c2HO(SP), AX; INCQ AX; MOVQ AX, c2HO(SP); JMP c2_loop_ho

c2_next_oc:
	MOVQ c2OC(SP), AX; INCQ AX; MOVQ AX, c2OC(SP); JMP c2_loop_oc

c2_next_g:
	MOVQ c2GRP(SP), AX; INCQ AX; MOVQ AX, c2GRP(SP); JMP c2_loop_g

c2_next_ni:
	MOVQ c2NI(SP), AX; INCQ AX; MOVQ AX, c2NI(SP); JMP c2_loop_ni

c2_done:
	VZEROUPPER
	RET

// conv2dSSE2 — full conv2d forward pass using SSE2.
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
	HADDPD X8, X8; ADDSD X8, X0
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
