#include "textflag.h"

// convTranspose2dNEON — full transposed-conv2d forward pass using ARM NEON/FP.
//
// func convTranspose2dNEON(out, x, wt, bias []float64,
//                           n, inC, h, w, outC, kH, kW, sH, sW, groups int)
//
// Arg offsets:
//   out:0 x:24 wt:48 bias:72
//   n:96 inC:104 h:112 w:120 outC:128 kH:136 kW:144 sH:152 sW:160 groups:168
//   Total: 176 bytes
//
// Frame: 512 bytes
#define ctnOUTP    0
#define ctnXP      8
#define ctnWTP     16
#define ctnBP      24
#define ctnN       32
#define ctnINC     40
#define ctnH       48
#define ctnW       56
#define ctnOUTC    64
#define ctnKH      72
#define ctnKW      80
#define ctnSH      88
#define ctnSW      96
#define ctnGROUPS  104
#define ctnHOUT    112
#define ctnWOUT    120
#define ctnICPERG  128
#define ctnOCPERG  136
#define ctnNI      144
#define ctnGRP     152
#define ctnIC      160
#define ctnHI      168
#define ctnWI      176
#define ctnOC      184
#define ctnKH_I    192
#define ctnICABS   200
#define ctnOCABS   208
#define ctnXVAL    216
#define ctnWROWP   224
#define ctnDSTP    232
#define ctnOCST    240
#define ctnICST    248

TEXT ·convTranspose2dNEON(SB), $512-176
	MOVD out+0(FP), R0;    MOVD R0, ctnOUTP(RSP)
	MOVD x+24(FP), R0;     MOVD R0, ctnXP(RSP)
	MOVD wt+48(FP), R0;    MOVD R0, ctnWTP(RSP)
	MOVD bias+72(FP), R0;  MOVD R0, ctnBP(RSP)
	MOVD n+96(FP), R0;     MOVD R0, ctnN(RSP)
	MOVD inC+104(FP), R0;  MOVD R0, ctnINC(RSP)
	MOVD h+112(FP), R0;    MOVD R0, ctnH(RSP)
	MOVD w+120(FP), R0;    MOVD R0, ctnW(RSP)
	MOVD outC+128(FP), R0; MOVD R0, ctnOUTC(RSP)
	MOVD kH+136(FP), R0;   MOVD R0, ctnKH(RSP)
	MOVD kW+144(FP), R0;   MOVD R0, ctnKW(RSP)
	MOVD sH+152(FP), R0;   MOVD R0, ctnSH(RSP)
	MOVD sW+160(FP), R0;   MOVD R0, ctnSW(RSP)
	MOVD groups+168(FP), R0; MOVD R0, ctnGROUPS(RSP)

	// hOut = (h-1)*sH + kH
	MOVD ctnH(RSP), R0; SUB $1, R0, R0; MOVD ctnSH(RSP), R1; MUL R1, R0, R0
	MOVD ctnKH(RSP), R1; ADD R1, R0, R0; MOVD R0, ctnHOUT(RSP)

	// wOut = (w-1)*sW + kW
	MOVD ctnW(RSP), R0; SUB $1, R0, R0; MOVD ctnSW(RSP), R1; MUL R1, R0, R0
	MOVD ctnKW(RSP), R1; ADD R1, R0, R0; MOVD R0, ctnWOUT(RSP)

	// icPerGroup = inC / groups
	MOVD ctnINC(RSP), R0; MOVD ctnGROUPS(RSP), R1; UDIV R1, R0, R2; MOVD R2, ctnICPERG(RSP)
	// ocPerGroup = outC / groups
	MOVD ctnOUTC(RSP), R0; MOVD ctnGROUPS(RSP), R1; UDIV R1, R0, R2; MOVD R2, ctnOCPERG(RSP)

	// Init output to bias
	MOVD $0, R0; MOVD R0, ctnNI(RSP)
ctn_init_ni:
	MOVD ctnNI(RSP), R0; MOVD ctnN(RSP), R1; CMP R1, R0; BGE ctn_init_done
	MOVD $0, R0; MOVD R0, ctnOC(RSP)
ctn_init_oc:
	MOVD ctnOC(RSP), R0; MOVD ctnOUTC(RSP), R1; CMP R1, R0; BGE ctn_init_next_ni
	MOVD ctnBP(RSP), R0; MOVD ctnOC(RSP), R1; LSL $3, R1, R1; ADD R1, R0, R0
	FMOVD (R0), F15
	MOVD ctnNI(RSP), R1; MOVD ctnOUTC(RSP), R2; MUL R2, R1, R1; MOVD ctnOC(RSP), R2; ADD R2, R1, R1
	MOVD ctnHOUT(RSP), R2; MUL R2, R1, R1; MOVD ctnWOUT(RSP), R2; MUL R2, R1, R1
	LSL $3, R1, R1; MOVD ctnOUTP(RSP), R2; ADD R2, R1, R1  // R1 = &out[base]
	MOVD ctnHOUT(RSP), R2; MOVD ctnWOUT(RSP), R3; MUL R3, R2, R2  // count
ctn_fill:
	CBZ R2, ctn_init_next_oc
	FMOVD.P F15, 8(R1)
	SUB $1, R2, R2; B ctn_fill
ctn_init_next_oc:
	MOVD ctnOC(RSP), R0; ADD $1, R0, R0; MOVD R0, ctnOC(RSP); B ctn_init_oc
ctn_init_next_ni:
	MOVD ctnNI(RSP), R0; ADD $1, R0, R0; MOVD R0, ctnNI(RSP); B ctn_init_ni
ctn_init_done:

	MOVD $0, R0; MOVD R0, ctnNI(RSP)
ctn_ni:
	MOVD ctnNI(RSP), R0; MOVD ctnN(RSP), R1; CMP R1, R0; BGE ctn_done
	MOVD $0, R0; MOVD R0, ctnGRP(RSP)
ctn_g:
	MOVD ctnGRP(RSP), R0; MOVD ctnGROUPS(RSP), R1; CMP R1, R0; BGE ctn_next_ni
	MOVD ctnGRP(RSP), R0; MOVD ctnOCPERG(RSP), R1; MUL R1, R0, R0; MOVD R0, ctnOCST(RSP)
	MOVD ctnGRP(RSP), R0; MOVD ctnICPERG(RSP), R1; MUL R1, R0, R0; MOVD R0, ctnICST(RSP)
	MOVD $0, R0; MOVD R0, ctnIC(RSP)
ctn_ic:
	MOVD ctnIC(RSP), R0; MOVD ctnICPERG(RSP), R1; CMP R1, R0; BGE ctn_next_g
	MOVD ctnICST(RSP), R0; MOVD ctnIC(RSP), R1; ADD R1, R0, R0; MOVD R0, ctnICABS(RSP)
	MOVD $0, R0; MOVD R0, ctnHI(RSP)
ctn_hi:
	MOVD ctnHI(RSP), R0; MOVD ctnH(RSP), R1; CMP R1, R0; BGE ctn_next_ic
	MOVD $0, R0; MOVD R0, ctnWI(RSP)
ctn_wi:
	MOVD ctnWI(RSP), R0; MOVD ctnW(RSP), R1; CMP R1, R0; BGE ctn_next_hi

	// xVal = x[(ni*inC + absIC)*h*w + hi*w + wi]
	MOVD ctnNI(RSP), R0; MOVD ctnINC(RSP), R1; MUL R1, R0, R0; MOVD ctnICABS(RSP), R1; ADD R1, R0, R0
	MOVD ctnH(RSP), R1; MUL R1, R0, R0; MOVD ctnW(RSP), R1; MUL R1, R0, R0
	MOVD ctnHI(RSP), R1; MOVD ctnW(RSP), R2; MUL R2, R1, R1; ADD R1, R0, R0; MOVD ctnWI(RSP), R1; ADD R1, R0, R0
	LSL $3, R0, R0; MOVD ctnXP(RSP), R1; ADD R1, R0, R0
	FMOVD (R0), F13; FMOVD F13, ctnXVAL(RSP)

	MOVD $0, R0; MOVD R0, ctnOC(RSP)
ctn_oc:
	MOVD ctnOC(RSP), R0; MOVD ctnOCPERG(RSP), R1; CMP R1, R0; BGE ctn_next_wi
	MOVD ctnOCST(RSP), R0; MOVD ctnOC(RSP), R1; ADD R1, R0, R0; MOVD R0, ctnOCABS(RSP)
	MOVD $0, R0; MOVD R0, ctnKH_I(RSP)
ctn_kh:
	MOVD ctnKH_I(RSP), R0; MOVD ctnKH(RSP), R1; CMP R1, R0; BGE ctn_next_oc

	// ho = hi*sH + kh
	MOVD ctnHI(RSP), R0; MOVD ctnSH(RSP), R1; MUL R1, R0, R0; MOVD ctnKH_I(RSP), R1; ADD R1, R0, R0

	// dstBase = (ni*outC + absOC)*hOut*wOut + ho*wOut + wi*sW
	MOVD ctnNI(RSP), R1; MOVD ctnOUTC(RSP), R2; MUL R2, R1, R1; MOVD ctnOCABS(RSP), R2; ADD R2, R1, R1
	MOVD ctnHOUT(RSP), R2; MUL R2, R1, R1; MOVD ctnWOUT(RSP), R2; MUL R2, R1, R1
	MOVD R0, R2; MOVD ctnWOUT(RSP), R3; MUL R3, R2, R2; ADD R2, R1, R1
	MOVD ctnWI(RSP), R2; MOVD ctnSW(RSP), R3; MUL R3, R2, R2; ADD R2, R1, R1
	LSL $3, R1, R1; MOVD ctnOUTP(RSP), R2; ADD R2, R1, R1; MOVD R1, ctnDSTP(RSP)

	// wRowPtr: (absIC*ocPerGroup*kH*kW + oc*kH*kW + kh*kW) * 8 + wtp
	MOVD ctnICABS(RSP), R0; MOVD ctnOCPERG(RSP), R1; MUL R1, R0, R0
	MOVD ctnKH(RSP), R1; MUL R1, R0, R0; MOVD ctnKW(RSP), R1; MUL R1, R0, R0
	MOVD ctnOC(RSP), R1; MOVD ctnKH(RSP), R2; MUL R2, R1, R1; MOVD ctnKW(RSP), R2; MUL R2, R1, R1; ADD R1, R0, R0
	MOVD ctnKH_I(RSP), R1; MOVD ctnKW(RSP), R2; MUL R2, R1, R1; ADD R1, R0, R0
	LSL $3, R0, R0; MOVD ctnWTP(RSP), R1; ADD R1, R0, R0; MOVD R0, ctnWROWP(RSP)

	// dst[0..kW-1] += xVal * wRow[0..kW-1]  (NEON FMADDD 4-unrolled)
	MOVD ctnDSTP(RSP), R3      // dst ptr
	MOVD ctnWROWP(RSP), R4     // weight ptr
	MOVD ctnKW(RSP), R5        // count
	FMOVD ctnXVAL(RSP), F13
ctn_fma4:
	CMP $4, R5; BLT ctn_fma_tail
	FMOVD.P 8(R4), F0; FMOVD (R3), F1; FMADDD F13, F0, F1, F1; FMOVD.P F1, 8(R3)
	FMOVD.P 8(R4), F2; FMOVD (R3), F3; FMADDD F13, F2, F3, F3; FMOVD.P F3, 8(R3)
	FMOVD.P 8(R4), F4; FMOVD (R3), F5; FMADDD F13, F4, F5, F5; FMOVD.P F5, 8(R3)
	FMOVD.P 8(R4), F6; FMOVD (R3), F7; FMADDD F13, F6, F7, F7; FMOVD.P F7, 8(R3)
	SUB $4, R5, R5; B ctn_fma4
ctn_fma_tail:
	CBZ R5, ctn_kh_next
ctn_fma_scalar:
	FMOVD.P 8(R4), F0; FMOVD (R3), F1; FMADDD F13, F0, F1, F1; FMOVD.P F1, 8(R3)
	SUBS $1, R5, R5; BNE ctn_fma_scalar

ctn_kh_next:
	MOVD ctnKH_I(RSP), R0; ADD $1, R0, R0; MOVD R0, ctnKH_I(RSP); B ctn_kh
ctn_next_oc:
	MOVD ctnOC(RSP), R0; ADD $1, R0, R0; MOVD R0, ctnOC(RSP); B ctn_oc
ctn_next_wi:
	MOVD ctnWI(RSP), R0; ADD $1, R0, R0; MOVD R0, ctnWI(RSP); B ctn_wi
ctn_next_hi:
	MOVD ctnHI(RSP), R0; ADD $1, R0, R0; MOVD R0, ctnHI(RSP); B ctn_hi
ctn_next_ic:
	MOVD ctnIC(RSP), R0; ADD $1, R0, R0; MOVD R0, ctnIC(RSP); B ctn_ic
ctn_next_g:
	MOVD ctnGRP(RSP), R0; ADD $1, R0, R0; MOVD R0, ctnGRP(RSP); B ctn_g
ctn_next_ni:
	MOVD ctnNI(RSP), R0; ADD $1, R0, R0; MOVD R0, ctnNI(RSP); B ctn_ni
ctn_done:
	RET
