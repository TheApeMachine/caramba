#include "textflag.h"

// convTranspose2dSSE2 — full transposed conv2d forward pass using SSE2.
//
// func convTranspose2dSSE2(out, x, wt, bias []float64,
//                           n, inC, h, w, outC, kH, kW, sH, sW, groups int)
//
// Arg offsets from FP (4 slices × 24 bytes = 96, then 10 ints × 8 = 80):
//   out:    0(FP)
//   x:     24(FP)
//   wt:    48(FP)
//   bias:  72(FP)
//   n:     96(FP)
//   inC:  104(FP)
//   h:    112(FP)
//   w:    120(FP)
//   outC: 128(FP)
//   kH:   136(FP)
//   kW:   144(FP)
//   sH:   152(FP)
//   sW:   160(FP)
//   groups:168(FP)
//   Total args: 176 bytes
//
// Algorithm (scatter-add):
//   hOut = (h-1)*sH + kH
//   wOut = (w-1)*sW + kW
//   Initialize out to bias values.
//   for ni,g,ic,hi,wi,oc,kh:
//     xVal = x[ni*inC*h*w + absIC*h*w + hi*w + wi]
//     dst  = out[ni*outC*hOut*wOut + absOC*hOut*wOut + (hi*sH+kh)*wOut + wi*sW]
//     wRow = wt[absIC*ocPerGroup*kH*kW + oc*kH*kW + kh*kW]  (kW elements)
//
// Frame: 512 bytes for spills.
#define ctAOUTP    0
#define ctAXP      8
#define ctAWTP     16
#define ctABP      24
#define ctAN       32
#define ctAINC     40
#define ctAH       48
#define ctAW       56
#define ctAOUTC    64
#define ctAKH      72
#define ctAKW      80
#define ctASH      88
#define ctASW      96
#define ctAGROUPS  104
#define ctAHOUT    112
#define ctAWOUT    120
#define ctAICPERG  128
#define ctAOCPERG  136
#define ctANI      144
#define ctAGRP     152
#define ctAIC      160
#define ctAHI      168
#define ctAWI      176
#define ctAOC      184
#define ctAKH_I    192
#define ctAICABS   200
#define ctAOCABS   208
#define ctAXVAL    216
#define ctAWROWP   224
#define ctADSTP    232
#define ctAOCST    240
#define ctAICST    248

TEXT ·convTranspose2dSSE2(SB), $512-176
	MOVQ out+0(FP), AX;    MOVQ AX, ctAOUTP(SP)
	MOVQ x+24(FP), AX;     MOVQ AX, ctAXP(SP)
	MOVQ wt+48(FP), AX;    MOVQ AX, ctAWTP(SP)
	MOVQ bias+72(FP), AX;  MOVQ AX, ctABP(SP)
	MOVQ n+96(FP), AX;     MOVQ AX, ctAN(SP)
	MOVQ inC+104(FP), AX;  MOVQ AX, ctAINC(SP)
	MOVQ h+112(FP), AX;    MOVQ AX, ctAH(SP)
	MOVQ w+120(FP), AX;    MOVQ AX, ctAW(SP)
	MOVQ outC+128(FP), AX; MOVQ AX, ctAOUTC(SP)
	MOVQ kH+136(FP), AX;   MOVQ AX, ctAKH(SP)
	MOVQ kW+144(FP), AX;   MOVQ AX, ctAKW(SP)
	MOVQ sH+152(FP), AX;   MOVQ AX, ctASH(SP)
	MOVQ sW+160(FP), AX;   MOVQ AX, ctASW(SP)
	MOVQ groups+168(FP), AX; MOVQ AX, ctAGROUPS(SP)

	MOVQ ctAH(SP), AX; DECQ AX; IMULQ ctASH(SP), AX; ADDQ ctAKH(SP), AX; MOVQ AX, ctAHOUT(SP)
	MOVQ ctAW(SP), AX; DECQ AX; IMULQ ctASW(SP), AX; ADDQ ctAKW(SP), AX; MOVQ AX, ctAWOUT(SP)
	MOVQ ctAINC(SP), AX; XORQ DX, DX; DIVQ ctAGROUPS(SP); MOVQ AX, ctAICPERG(SP)
	MOVQ ctAOUTC(SP), AX; XORQ DX, DX; DIVQ ctAGROUPS(SP); MOVQ AX, ctAOCPERG(SP)

	// Init bias
	MOVQ $0, BX; MOVQ BX, ctANI(SP)
ctS_init_ni:
	MOVQ ctANI(SP), BX; CMPQ BX, ctAN(SP); JGE ctS_init_done
	MOVQ $0, BX; MOVQ BX, ctAOC(SP)
ctS_init_oc:
	MOVQ ctAOC(SP), BX; CMPQ BX, ctAOUTC(SP); JGE ctS_init_next_ni
	MOVQ ctABP(SP), AX; MOVQ ctAOC(SP), DX; SHLQ $3, DX; ADDQ DX, AX; MOVSD (AX), X15
	MOVQ ctANI(SP), DX; IMULQ ctAOUTC(SP), DX; ADDQ ctAOC(SP), DX
	MOVQ ctAHOUT(SP), CX; IMULQ ctAWOUT(SP), CX; IMULQ CX, DX
	SHLQ $3, DX; MOVQ ctAOUTP(SP), DI; ADDQ DX, DI
	MOVQ ctAHOUT(SP), DX; IMULQ ctAWOUT(SP), DX
	UNPCKLPD X15, X15
ctS_fill:
	CMPQ DX, $2; JL ctS_fill_s; MOVUPD X15, (DI); ADDQ $16, DI; SUBQ $2, DX; JMP ctS_fill
ctS_fill_s:
	CMPQ DX, $0; JLE ctS_init_next_oc; MOVSD X15, (DI)
ctS_init_next_oc:
	MOVQ ctAOC(SP), AX; INCQ AX; MOVQ AX, ctAOC(SP); JMP ctS_init_oc
ctS_init_next_ni:
	MOVQ ctANI(SP), AX; INCQ AX; MOVQ AX, ctANI(SP); JMP ctS_init_ni
ctS_init_done:

	MOVQ $0, BX; MOVQ BX, ctANI(SP)
ctS_ni:
	MOVQ ctANI(SP), BX; CMPQ BX, ctAN(SP); JGE ctS_done
	MOVQ $0, BX; MOVQ BX, ctAGRP(SP)
ctS_g:
	MOVQ ctAGRP(SP), BX; CMPQ BX, ctAGROUPS(SP); JGE ctS_next_ni
	MOVQ ctAGRP(SP), AX; IMULQ ctAOCPERG(SP), AX; MOVQ AX, ctAOCST(SP)
	MOVQ ctAGRP(SP), AX; IMULQ ctAICPERG(SP), AX; MOVQ AX, ctAICST(SP)
	MOVQ $0, BX; MOVQ BX, ctAIC(SP)
ctS_ic:
	MOVQ ctAIC(SP), BX; CMPQ BX, ctAICPERG(SP); JGE ctS_next_g
	MOVQ ctAICST(SP), AX; ADDQ ctAIC(SP), AX; MOVQ AX, ctAICABS(SP)
	MOVQ $0, BX; MOVQ BX, ctAHI(SP)
ctS_hi:
	MOVQ ctAHI(SP), BX; CMPQ BX, ctAH(SP); JGE ctS_next_ic
	MOVQ $0, BX; MOVQ BX, ctAWI(SP)
ctS_wi:
	MOVQ ctAWI(SP), BX; CMPQ BX, ctAW(SP); JGE ctS_next_hi
	MOVQ ctANI(SP), AX; IMULQ ctAINC(SP), AX; ADDQ ctAICABS(SP), AX
	MOVQ ctAH(SP), DX; IMULQ ctAW(SP), DX; IMULQ DX, AX
	MOVQ ctAHI(SP), DX; IMULQ ctAW(SP), DX; ADDQ DX, AX; ADDQ ctAWI(SP), AX
	SHLQ $3, AX; MOVQ ctAXP(SP), DX; ADDQ DX, AX; MOVSD (AX), X13; MOVSD X13, ctAXVAL(SP)
	UNPCKLPD X13, X13  // X13 = xVal in both lanes
	MOVQ $0, BX; MOVQ BX, ctAOC(SP)
ctS_oc:
	MOVQ ctAOC(SP), BX; CMPQ BX, ctAOCPERG(SP); JGE ctS_next_wi
	MOVQ ctAOCST(SP), AX; ADDQ ctAOC(SP), AX; MOVQ AX, ctAOCABS(SP)
	MOVQ $0, BX; MOVQ BX, ctAKH_I(SP)
ctS_kh:
	MOVQ ctAKH_I(SP), BX; CMPQ BX, ctAKH(SP); JGE ctS_next_oc
	MOVQ ctAHI(SP), AX; IMULQ ctASH(SP), AX; ADDQ ctAKH_I(SP), AX
	MOVQ ctANI(SP), DX; IMULQ ctAOUTC(SP), DX; ADDQ ctAOCABS(SP), DX
	MOVQ ctAHOUT(SP), CX; IMULQ ctAWOUT(SP), CX; IMULQ CX, DX
	MOVQ AX, CX; IMULQ ctAWOUT(SP), CX; ADDQ CX, DX
	MOVQ ctAWI(SP), CX; IMULQ ctASW(SP), CX; ADDQ CX, DX
	SHLQ $3, DX; MOVQ ctAOUTP(SP), CX; ADDQ CX, DX; MOVQ DX, ctADSTP(SP)
	MOVQ ctAICABS(SP), AX; IMULQ ctAOCPERG(SP), AX; IMULQ ctAKH(SP), AX; IMULQ ctAKW(SP), AX
	MOVQ ctAOC(SP), DX; MOVQ ctAKH(SP), CX; IMULQ CX, DX; IMULQ ctAKW(SP), DX; ADDQ DX, AX
	MOVQ ctAKH_I(SP), DX; IMULQ ctAKW(SP), DX; ADDQ DX, AX
	SHLQ $3, AX; MOVQ ctAWTP(SP), DX; ADDQ DX, AX; MOVQ AX, ctAWROWP(SP)
	MOVQ ctADSTP(SP), DI; MOVQ ctAWROWP(SP), SI; MOVQ ctAKW(SP), CX
ctS_inner:
	CMPQ CX, $2; JL ctS_scalar
	MOVUPD (DI), X0; MOVUPD (SI), X1; MULPD X13, X1; ADDPD X1, X0; MOVUPD X0, (DI)
	ADDQ $16, DI; ADDQ $16, SI; SUBQ $2, CX; JMP ctS_inner
ctS_scalar:
	CMPQ CX, $0; JLE ctS_kh_next
	MOVSD ctAXVAL(SP), X13; MOVSD (SI), X1; MULSD X13, X1; MOVSD (DI), X2; ADDSD X2, X1; MOVSD X1, (DI)
ctS_kh_next:
	MOVQ ctAKH_I(SP), AX; INCQ AX; MOVQ AX, ctAKH_I(SP); JMP ctS_kh
ctS_next_oc:
	MOVQ ctAOC(SP), AX; INCQ AX; MOVQ AX, ctAOC(SP); JMP ctS_oc
ctS_next_wi:
	MOVQ ctAWI(SP), AX; INCQ AX; MOVQ AX, ctAWI(SP); JMP ctS_wi
ctS_next_hi:
	MOVQ ctAHI(SP), AX; INCQ AX; MOVQ AX, ctAHI(SP); JMP ctS_hi
ctS_next_ic:
	MOVQ ctAIC(SP), AX; INCQ AX; MOVQ AX, ctAIC(SP); JMP ctS_ic
ctS_next_g:
	MOVQ ctAGRP(SP), AX; INCQ AX; MOVQ AX, ctAGRP(SP); JMP ctS_g
ctS_next_ni:
	MOVQ ctANI(SP), AX; INCQ AX; MOVQ AX, ctANI(SP); JMP ctS_ni
ctS_done:
	RET
