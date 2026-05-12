#include "textflag.h"

// convTranspose2dAVX2 — full transposed-conv2d forward pass using AVX2+FMA.
//
// func convTranspose2dAVX2(out, x, wt, bias []float64,
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
//     dst[0..kW-1] += xVal * wRow[0..kW-1]   (AVX2 FMA, 4-wide)
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

TEXT ·convTranspose2dAVX2(SB), $512-176
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

	// hOut = (h-1)*sH + kH
	MOVQ ctAH(SP), AX; DECQ AX
	IMULQ ctASH(SP), AX
	ADDQ ctAKH(SP), AX
	MOVQ AX, ctAHOUT(SP)

	// wOut = (w-1)*sW + kW
	MOVQ ctAW(SP), AX; DECQ AX
	IMULQ ctASW(SP), AX
	ADDQ ctAKW(SP), AX
	MOVQ AX, ctAWOUT(SP)

	// icPerGroup = inC / groups
	MOVQ ctAINC(SP), AX; XORQ DX, DX; DIVQ ctAGROUPS(SP)
	MOVQ AX, ctAICPERG(SP)

	// ocPerGroup = outC / groups
	MOVQ ctAOUTC(SP), AX; XORQ DX, DX; DIVQ ctAGROUPS(SP)
	MOVQ AX, ctAOCPERG(SP)

	// Initialize out to bias: for ni in [0,n), for oc in [0,outC): fill hOut*wOut with bias[oc]
	MOVQ $0, BX            // ni
ctA_init_ni:
	CMPQ BX, ctAN(SP); JGE ctA_init_done
	MOVQ $0, CX            // oc
ctA_init_oc:
	CMPQ CX, ctAOUTC(SP); JGE ctA_init_next_ni

	// load bias[oc]
	MOVQ ctABP(SP), AX
	MOVQ CX, DX; SHLQ $3, DX; ADDQ DX, AX
	VMOVSD (AX), X15
	VBROADCASTSD X15, Y15

	// base = (ni*outC + oc) * hOut*wOut
	MOVQ BX, AX; IMULQ ctAOUTC(SP), AX; ADDQ CX, AX
	MOVQ ctAHOUT(SP), DX; IMULQ ctAWOUT(SP), DX
	IMULQ DX, AX
	SHLQ $3, AX
	MOVQ ctAOUTP(SP), DI; ADDQ AX, DI   // DI = &out[base]

	// count = hOut*wOut
	MOVQ ctAHOUT(SP), DX; IMULQ ctAWOUT(SP), DX
	// fill 4 at a time
ctA_fill_loop:
	CMPQ DX, $4; JL ctA_fill_tail
	VMOVUPD Y15, (DI); ADDQ $32, DI; SUBQ $4, DX; JMP ctA_fill_loop
ctA_fill_tail:
	CMPQ DX, $0; JLE ctA_init_next_oc
	VMOVSD X15, (DI); ADDQ $8, DI; DECQ DX; JMP ctA_fill_tail
ctA_init_next_oc:
	INCQ CX; JMP ctA_init_oc
ctA_init_next_ni:
	INCQ BX; JMP ctA_init_ni
ctA_init_done:

	// Main scatter-add loops
	MOVQ $0, BX; MOVQ BX, ctANI(SP)
ctA_ni:
	MOVQ ctANI(SP), BX; CMPQ BX, ctAN(SP); JGE ctA_done

	MOVQ $0, BX; MOVQ BX, ctAGRP(SP)
ctA_g:
	MOVQ ctAGRP(SP), BX; CMPQ BX, ctAGROUPS(SP); JGE ctA_next_ni

	// ocStart = g*ocPerGroup
	MOVQ ctAGRP(SP), AX; IMULQ ctAOCPERG(SP), AX; MOVQ AX, ctAOCST(SP)
	// icStart = g*icPerGroup
	MOVQ ctAGRP(SP), AX; IMULQ ctAICPERG(SP), AX; MOVQ AX, ctAICST(SP)

	MOVQ $0, BX; MOVQ BX, ctAIC(SP)
ctA_ic:
	MOVQ ctAIC(SP), BX; CMPQ BX, ctAICPERG(SP); JGE ctA_next_g

	// absIC = icStart + ic
	MOVQ ctAICST(SP), AX; ADDQ ctAIC(SP), AX; MOVQ AX, ctAICABS(SP)

	// wRowBase for this ic: absIC * ocPerGroup*kH*kW  (element index)
	MOVQ ctAICABS(SP), AX
	MOVQ ctAOCPERG(SP), DX; IMULQ DX, AX
	MOVQ ctAKH(SP), DX; IMULQ DX, AX
	MOVQ ctAKW(SP), DX; IMULQ DX, AX
	// AX = element index; save as byte offset later per (oc,kh)

	MOVQ $0, BX; MOVQ BX, ctAHI(SP)
ctA_hi:
	MOVQ ctAHI(SP), BX; CMPQ BX, ctAH(SP); JGE ctA_next_ic

	MOVQ $0, BX; MOVQ BX, ctAWI(SP)
ctA_wi:
	MOVQ ctAWI(SP), BX; CMPQ BX, ctAW(SP); JGE ctA_next_hi

	// xVal = x[ni*inC*h*w + absIC*h*w + hi*w + wi]
	MOVQ ctANI(SP), AX; IMULQ ctAINC(SP), AX
	ADDQ ctAICABS(SP), AX
	IMULQ ctAH(SP), AX; IMULQ ctAW(SP), AX   // wait — wrong: (ni*inC+absIC)*h*w
	// Fix: idx = (ni*inC + absIC)*h*w + hi*w + wi
	MOVQ ctANI(SP), AX; IMULQ ctAINC(SP), AX; ADDQ ctAICABS(SP), AX
	MOVQ ctAH(SP), DX; IMULQ ctAW(SP), DX; IMULQ DX, AX
	MOVQ ctAHI(SP), DX; IMULQ ctAW(SP), DX; ADDQ DX, AX
	ADDQ ctAWI(SP), AX
	SHLQ $3, AX; MOVQ ctAXP(SP), DX; ADDQ DX, AX
	VMOVSD (AX), X14
	VMOVSD X14, ctAXVAL(SP)
	VBROADCASTSD X14, Y14   // Y14 = xVal broadcast

	MOVQ $0, BX; MOVQ BX, ctAOC(SP)
ctA_oc:
	MOVQ ctAOC(SP), BX; CMPQ BX, ctAOCPERG(SP); JGE ctA_next_wi

	// absOC = ocStart + oc
	MOVQ ctAOCST(SP), AX; ADDQ ctAOC(SP), AX; MOVQ AX, ctAOCABS(SP)

	MOVQ $0, BX; MOVQ BX, ctAKH_I(SP)
ctA_kh:
	MOVQ ctAKH_I(SP), BX; CMPQ BX, ctAKH(SP); JGE ctA_next_oc

	// ho = hi*sH + kh
	MOVQ ctAHI(SP), AX; IMULQ ctASH(SP), AX; ADDQ ctAKH_I(SP), AX

	// dstBase = (ni*outC + absOC)*hOut*wOut + ho*wOut + wi*sW
	MOVQ ctANI(SP), DX; IMULQ ctAOUTC(SP), DX; ADDQ ctAOCABS(SP), DX
	MOVQ ctAHOUT(SP), CX; IMULQ ctAWOUT(SP), CX; IMULQ CX, DX
	MOVQ AX, CX; IMULQ ctAWOUT(SP), CX; ADDQ CX, DX
	MOVQ ctAWI(SP), CX; IMULQ ctASW(SP), CX; ADDQ CX, DX
	SHLQ $3, DX; MOVQ ctAOUTP(SP), CX; ADDQ CX, DX
	MOVQ DX, ctADSTP(SP)

	// wRowPtr: (absIC*ocPerGroup*kH*kW + oc*kH*kW + kh*kW) * 8 + wtp
	MOVQ ctAICABS(SP), AX; IMULQ ctAOCPERG(SP), AX
	MOVQ ctAKH(SP), DX; IMULQ DX, AX
	MOVQ ctAKW(SP), DX; IMULQ DX, AX
	MOVQ ctAOC(SP), DX; MOVQ ctAKH(SP), CX; IMULQ CX, DX; MOVQ ctAKW(SP), CX; IMULQ CX, DX; ADDQ DX, AX
	MOVQ ctAKH_I(SP), DX; IMULQ ctAKW(SP), DX; ADDQ DX, AX
	SHLQ $3, AX; MOVQ ctAWTP(SP), DX; ADDQ DX, AX
	MOVQ AX, ctAWROWP(SP)

	// dst[0..kW-1] += xVal * wRow[0..kW-1]   (AVX2 FMA 4-wide)
	MOVQ ctADSTP(SP), DI     // dst ptr
	MOVQ ctAWROWP(SP), SI    // weight ptr
	MOVQ ctAKW(SP), CX       // count
ctA_fma:
	CMPQ CX, $4; JL ctA_fma_tail
	VMOVUPD (DI), Y0
	VMOVUPD (SI), Y1
	VFMADD231PD Y14, Y1, Y0
	VMOVUPD Y0, (DI)
	ADDQ $32, DI; ADDQ $32, SI; SUBQ $4, CX; JMP ctA_fma
ctA_fma_tail:
	CMPQ CX, $2; JL ctA_fma_scalar
	MOVUPD (DI), X0
	MOVUPD (SI), X1
	MULPD X14, X1
	ADDPD X1, X0
	MOVUPD X0, (DI)
	ADDQ $16, DI; ADDQ $16, SI; SUBQ $2, CX
ctA_fma_scalar:
	CMPQ CX, $0; JLE ctA_kh_next
	VMOVSD ctAXVAL(SP), X14
	MOVSD  (SI), X1
	MULSD  X14, X1
	MOVSD  (DI), X2
	ADDSD  X2, X1
	MOVSD  X1, (DI)
ctA_kh_next:
	VZEROUPPER
	MOVQ ctAKH_I(SP), AX; INCQ AX; MOVQ AX, ctAKH_I(SP); JMP ctA_kh

ctA_next_oc:
	MOVQ ctAOC(SP), AX; INCQ AX; MOVQ AX, ctAOC(SP); JMP ctA_oc
ctA_next_wi:
	MOVQ ctAWI(SP), AX; INCQ AX; MOVQ AX, ctAWI(SP); JMP ctA_wi
ctA_next_hi:
	MOVQ ctAHI(SP), AX; INCQ AX; MOVQ AX, ctAHI(SP); JMP ctA_hi
ctA_next_ic:
	MOVQ ctAIC(SP), AX; INCQ AX; MOVQ AX, ctAIC(SP); JMP ctA_ic
ctA_next_g:
	MOVQ ctAGRP(SP), AX; INCQ AX; MOVQ AX, ctAGRP(SP); JMP ctA_g
ctA_next_ni:
	MOVQ ctANI(SP), AX; INCQ AX; MOVQ AX, ctANI(SP); JMP ctA_ni
ctA_done:
	VZEROUPPER
	RET

// convTranspose2dSSE2 — same as AVX2 but uses SSE2 (2-wide MULPD/ADDPD).
//
// func convTranspose2dSSE2(out, x, wt, bias []float64,
//                           n, inC, h, w, outC, kH, kW, sH, sW, groups int)
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
