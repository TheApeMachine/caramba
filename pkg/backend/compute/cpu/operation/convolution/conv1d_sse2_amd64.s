#include "textflag.h"

// conv1dSSE2 — full conv1d forward pass using SSE2.
//
// func conv1dSSE2(out, x, weight, bias []float64,
//                 n, inC, l, outC, kSize, stride, pad, dilation, groups, lOut int)
//
// Arg offsets from FP (each slice = ptr+len+cap = 24 bytes):
//   out:      0(FP)  len:8   cap:16
//   x:       24(FP)  len:32  cap:40
//   weight:  48(FP)  len:56  cap:64
//   bias:    72(FP)  len:80  cap:88
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
// Frame slots (relative to SP after prologue, frame=256):
#define fOUTP    0
#define fXP      8
#define fWP      16
#define fBP      24
#define fN       32
#define fINC     40
#define fL       48
#define fOUTC    56
#define fKSIZE   64
#define fSTRIDE  72
#define fPAD     80
#define fDIL     88
#define fGROUPS  96
#define fLOUT    104
#define fNI      112   // loop: ni
#define fGRP     120   // loop: g
#define fOC      128   // loop: oc
#define fLO      136   // loop: lo
#define fIC      144   // loop: ic
#define fICPERG  152   // icPerGroup
#define fOCPERG  160   // ocPerGroup
#define fOCSTART 168   // ocStart
#define fICSSTART 176  // icStart
#define fWROWP   184   // weightRow ptr
#define fSUM     192   // current sum (float64)

TEXT ·conv1dSSE2(SB), $256-176
	MOVQ out+0(FP), AX
	MOVQ AX, fOUTP(SP)
	MOVQ x+24(FP), AX
	MOVQ AX, fXP(SP)
	MOVQ weight+48(FP), AX
	MOVQ AX, fWP(SP)
	MOVQ bias+72(FP), AX
	MOVQ AX, fBP(SP)
	MOVQ n+96(FP), AX
	MOVQ AX, fN(SP)
	MOVQ inC+104(FP), AX
	MOVQ AX, fINC(SP)
	MOVQ l+112(FP), AX
	MOVQ AX, fL(SP)
	MOVQ outC+120(FP), AX
	MOVQ AX, fOUTC(SP)
	MOVQ kSize+128(FP), AX
	MOVQ AX, fKSIZE(SP)
	MOVQ stride+136(FP), AX
	MOVQ AX, fSTRIDE(SP)
	MOVQ pad+144(FP), AX
	MOVQ AX, fPAD(SP)
	MOVQ dilation+152(FP), AX
	MOVQ AX, fDIL(SP)
	MOVQ groups+160(FP), AX
	MOVQ AX, fGROUPS(SP)
	MOVQ lOut+168(FP), AX
	MOVQ AX, fLOUT(SP)

	XORQ AX, AX
	MOVQ AX, fNI(SP)

sse2_loop_ni:
	MOVQ fNI(SP), AX
	MOVQ fN(SP), BX
	CMPQ AX, BX
	JGE  sse2_done

	MOVQ fINC(SP), AX
	MOVQ fGROUPS(SP), BX
	XORQ DX, DX
	DIVQ BX
	MOVQ AX, fICPERG(SP)

	MOVQ fOUTC(SP), AX
	MOVQ fGROUPS(SP), BX
	XORQ DX, DX
	DIVQ BX
	MOVQ AX, fOCPERG(SP)

	XORQ AX, AX
	MOVQ AX, fGRP(SP)

sse2_loop_g:
	MOVQ fGRP(SP), AX
	MOVQ fGROUPS(SP), BX
	CMPQ AX, BX
	JGE  sse2_next_ni

	MOVQ fGRP(SP), AX
	MOVQ fOCPERG(SP), BX
	IMULQ BX, AX
	MOVQ AX, fOCSTART(SP)

	MOVQ fGRP(SP), AX
	MOVQ fICPERG(SP), BX
	IMULQ BX, AX
	MOVQ AX, fICSSTART(SP)

	MOVQ fOCSTART(SP), AX
	MOVQ AX, fOC(SP)

sse2_loop_oc:
	MOVQ fOC(SP), AX
	MOVQ fOCSTART(SP), BX
	MOVQ fOCPERG(SP), CX
	ADDQ BX, CX
	CMPQ AX, CX
	JGE  sse2_next_g

	MOVQ fOC(SP), AX
	MOVQ fICPERG(SP), BX
	IMULQ BX, AX
	MOVQ fKSIZE(SP), BX
	IMULQ BX, AX
	SHLQ $3, AX
	MOVQ fWP(SP), BX
	ADDQ BX, AX
	MOVQ AX, fWROWP(SP)

	XORQ AX, AX
	MOVQ AX, fLO(SP)

sse2_loop_lo:
	MOVQ fLO(SP), AX
	MOVQ fLOUT(SP), BX
	CMPQ AX, BX
	JGE  sse2_next_oc

	MOVQ fBP(SP), AX
	MOVQ fOC(SP), BX
	SHLQ $3, BX
	MOVSD (AX)(BX*1), X15
	MOVSD X15, fSUM(SP)

	XORQ AX, AX
	MOVQ AX, fIC(SP)

sse2_loop_ic:
	MOVQ fIC(SP), AX
	MOVQ fICPERG(SP), BX
	CMPQ AX, BX
	JGE  sse2_write_out

	MOVQ fICSSTART(SP), AX
	MOVQ fIC(SP), BX
	ADDQ BX, AX
	MOVQ fNI(SP), BX
	MOVQ fINC(SP), CX
	IMULQ CX, BX
	ADDQ AX, BX
	MOVQ fL(SP), CX
	IMULQ CX, BX         // BX = inputBase

	MOVQ fIC(SP), DX
	MOVQ fKSIZE(SP), CX
	IMULQ CX, DX         // DX = weightBase

	MOVQ fDIL(SP), R8
	CMPQ R8, $1
	JNE  sse2_scalar_k

	MOVQ fLO(SP), R8
	MOVQ fSTRIDE(SP), R9
	IMULQ R9, R8
	MOVQ fPAD(SP), R9
	CMPQ R8, R9
	JL   sse2_scalar_k

	MOVQ R8, R10
	SUBQ R9, R10
	MOVQ fKSIZE(SP), R11
	ADDQ R11, R10
	SUBQ $1, R10
	MOVQ fL(SP), R11
	CMPQ R10, R11
	JGE  sse2_scalar_k

	MOVQ fLO(SP), R8
	MOVQ fSTRIDE(SP), R9
	IMULQ R9, R8
	MOVQ fPAD(SP), R9
	SUBQ R9, R8           // R8 = inputIdx for k=0

	MOVQ BX, R9
	ADDQ R8, R9
	SHLQ $3, R9
	MOVQ fXP(SP), R10
	ADDQ R10, R9

	MOVQ DX, R10
	SHLQ $3, R10
	MOVQ fWROWP(SP), R11
	ADDQ R11, R10

	MOVQ fKSIZE(SP), R11
	MOVSD fSUM(SP), X0
	XORPS X8, X8

sse2_k_loop:
	CMPQ R11, $2
	JL   sse2_k_tail
	MOVUPD 0(R9), X1
	MOVUPD 0(R10), X2
	MULPD  X2, X1
	ADDPD  X1, X8
	ADDQ $16, R9
	ADDQ $16, R10
	SUBQ $2, R11
	JMP  sse2_k_loop
sse2_k_tail:
	HADDPD X8, X8
	ADDSD  X8, X0
	CMPQ R11, $0
	JLE  sse2_k_done
sse2_k_scalar:
	MOVSD (R9), X1
	MOVSD (R10), X2
	MULSD X2, X1
	ADDSD X1, X0
	ADDQ $8, R9
	ADDQ $8, R10
	DECQ R11
	JNZ  sse2_k_scalar
sse2_k_done:
	MOVSD X0, fSUM(SP)
	JMP  sse2_next_ic

sse2_scalar_k:
	MOVQ fKSIZE(SP), R8
	XORQ R9, R9
	MOVSD fSUM(SP), X0
sse2_sk:
	CMPQ R9, R8
	JGE  sse2_sk_done

	MOVQ fLO(SP), R10
	MOVQ fSTRIDE(SP), R11
	IMULQ R11, R10
	MOVQ R9, R11
	MOVQ fDIL(SP), R12
	IMULQ R12, R11
	ADDQ R11, R10
	MOVQ fPAD(SP), R11
	SUBQ R11, R10

	CMPQ R10, $0
	JL   sse2_skip_k
	MOVQ fL(SP), R11
	CMPQ R10, R11
	JGE  sse2_skip_k

	MOVQ BX, R11
	ADDQ R10, R11
	SHLQ $3, R11
	MOVQ fXP(SP), R12
	MOVSD (R12)(R11*1), X1

	MOVQ DX, R11
	ADDQ R9, R11
	SHLQ $3, R11
	MOVQ fWROWP(SP), R12
	MOVSD (R12)(R11*1), X2
	MULSD X2, X1
	ADDSD X1, X0
sse2_skip_k:
	INCQ R9
	JMP  sse2_sk
sse2_sk_done:
	MOVSD X0, fSUM(SP)

sse2_next_ic:
	MOVQ fIC(SP), AX
	INCQ AX
	MOVQ AX, fIC(SP)
	JMP  sse2_loop_ic

sse2_write_out:
	MOVQ fNI(SP), AX
	MOVQ fOUTC(SP), BX
	IMULQ BX, AX
	MOVQ fLOUT(SP), BX
	IMULQ BX, AX
	MOVQ fOC(SP), BX
	MOVQ fLOUT(SP), CX
	IMULQ CX, BX
	ADDQ BX, AX
	ADDQ fLO(SP), AX
	SHLQ $3, AX
	MOVQ fOUTP(SP), BX
	MOVSD fSUM(SP), X0
	MOVSD X0, (BX)(AX*1)

	MOVQ fLO(SP), AX
	INCQ AX
	MOVQ AX, fLO(SP)
	JMP  sse2_loop_lo

sse2_next_oc:
	MOVQ fOC(SP), AX
	INCQ AX
	MOVQ AX, fOC(SP)
	JMP  sse2_loop_oc

sse2_next_g:
	MOVQ fGRP(SP), AX
	INCQ AX
	MOVQ AX, fGRP(SP)
	JMP  sse2_loop_g

sse2_next_ni:
	MOVQ fNI(SP), AX
	INCQ AX
	MOVQ AX, fNI(SP)
	JMP  sse2_loop_ni

sse2_done:
	RET
