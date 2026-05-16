#include "textflag.h"

// conv1dAVX2 — full conv1d forward pass using AVX2+FMA.
//
// func conv1dAVX2(out, x, weight, bias []float64,
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

TEXT ·conv1dAVX2(SB), $256-176
	// Load all args into frame.
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

	// ni = 0
	XORQ AX, AX
	MOVQ AX, fNI(SP)

loop_ni:
	MOVQ fNI(SP), AX
	MOVQ fN(SP), BX
	CMPQ AX, BX
	JGE  done_conv1d

	// Precompute icPerGroup = inC/groups, ocPerGroup = outC/groups
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

	// g = 0
	XORQ AX, AX
	MOVQ AX, fGRP(SP)

loop_g:
	MOVQ fGRP(SP), AX
	MOVQ fGROUPS(SP), BX
	CMPQ AX, BX
	JGE  next_ni

	// ocStart = g * ocPerGroup
	MOVQ fGRP(SP), AX
	MOVQ fOCPERG(SP), BX
	IMULQ BX, AX
	MOVQ AX, fOCSTART(SP)

	// icStart = g * icPerGroup
	MOVQ fGRP(SP), AX
	MOVQ fICPERG(SP), BX
	IMULQ BX, AX
	MOVQ AX, fICSSTART(SP)

	// oc = ocStart
	MOVQ fOCSTART(SP), AX
	MOVQ AX, fOC(SP)

loop_oc:
	// while oc < ocStart+ocPerGroup
	MOVQ fOC(SP), AX
	MOVQ fOCSTART(SP), BX
	MOVQ fOCPERG(SP), CX
	ADDQ BX, CX
	CMPQ AX, CX
	JGE  next_g

	// weightRow ptr = weight + oc*icPerGroup*kSize*8
	// weightRow = weight[oc*icPerGroup*kSize : ...]
	MOVQ fOC(SP), AX
	MOVQ fICPERG(SP), BX
	IMULQ BX, AX
	MOVQ fKSIZE(SP), BX
	IMULQ BX, AX        // AX = oc*icPerGroup*kSize
	SHLQ $3, AX         // *8 bytes
	MOVQ fWP(SP), BX
	ADDQ BX, AX
	MOVQ AX, fWROWP(SP)

	// lo = 0
	XORQ AX, AX
	MOVQ AX, fLO(SP)

loop_lo:
	MOVQ fLO(SP), AX
	MOVQ fLOUT(SP), BX
	CMPQ AX, BX
	JGE  next_oc

	// sum = bias[oc]
	MOVQ fBP(SP), AX
	MOVQ fOC(SP), BX
	SHLQ $3, BX
	MOVSD (AX)(BX*1), X15
	MOVSD X15, fSUM(SP)

	// ic = 0
	XORQ AX, AX
	MOVQ AX, fIC(SP)

loop_ic:
	MOVQ fIC(SP), AX
	MOVQ fICPERG(SP), BX
	CMPQ AX, BX
	JGE  write_out

	// absIC = icStart + ic
	MOVQ fICSSTART(SP), AX
	MOVQ fIC(SP), BX
	ADDQ BX, AX
	// inputBase = ni*inC*l + absIC*l
	// = (ni*inC + absIC) * l
	MOVQ fNI(SP), BX
	MOVQ fINC(SP), CX
	IMULQ CX, BX      // BX = ni*inC
	ADDQ AX, BX       // BX = ni*inC + absIC
	MOVQ fL(SP), CX
	IMULQ CX, BX      // BX = inputBase (element index)
	// weightBase = ic * kSize (element index into weightRow)
	MOVQ fIC(SP), DX
	MOVQ fKSIZE(SP), CX
	IMULQ CX, DX      // DX = ic*kSize

	// Check if this lo position is fully in-bounds for fast path:
	// fast path: dilation==1 AND lo*stride >= pad AND lo*stride+(kSize-1) < l+pad
	// i.e., inputIdx for k=0: lo*stride - pad >= 0
	//        inputIdx for k=kSize-1: lo*stride + (kSize-1) - pad < l
	MOVQ fDIL(SP), R8
	CMPQ R8, $1
	JNE  scalar_k_loop

	MOVQ fLO(SP), R8
	MOVQ fSTRIDE(SP), R9
	IMULQ R9, R8          // R8 = lo*stride
	MOVQ fPAD(SP), R9
	CMPQ R8, R9           // lo*stride >= pad?
	JL   scalar_k_loop

	// lo*stride - pad + kSize - 1 < l?
	MOVQ R8, R10
	SUBQ R9, R10          // R10 = lo*stride - pad
	MOVQ fKSIZE(SP), R11
	ADDQ R11, R10
	SUBQ $1, R10          // R10 = lo*stride - pad + kSize - 1
	MOVQ fL(SP), R11
	CMPQ R10, R11
	JGE  scalar_k_loop

	// Fast path: contiguous input slice, use AVX2 FMA.
	// inputIdx = lo*stride - pad (= R10 - kSize + 1... recalculate)
	MOVQ fLO(SP), R8
	MOVQ fSTRIDE(SP), R9
	IMULQ R9, R8
	MOVQ fPAD(SP), R9
	SUBQ R9, R8           // R8 = lo*stride - pad = inputIdx for k=0

	// xPtr = xP + (inputBase + R8)*8
	MOVQ BX, R9
	ADDQ R8, R9
	SHLQ $3, R9
	MOVQ fXP(SP), R10
	ADDQ R10, R9          // R9 = ptr to x[inputBase + inputIdx]

	// wPtr = wRowP + DX*8
	MOVQ DX, R10
	SHLQ $3, R10
	MOVQ fWROWP(SP), R11
	ADDQ R11, R10         // R10 = ptr to weightRow[ic*kSize]

	// kSize in R11
	MOVQ fKSIZE(SP), R11
	MOVSD fSUM(SP), X0

	// AVX2 4-wide FMA loop over kSize
	VXORPD Y1, Y1, Y1    // accumulator
avx2_k_loop:
	CMPQ R11, $4
	JL   avx2_k_tail
	VMOVUPD  0(R9), Y2
	VMOVUPD  0(R10), Y3
	VFMADD231PD Y2, Y3, Y1
	ADDQ $32, R9
	ADDQ $32, R10
	SUBQ $4, R11
	JMP  avx2_k_loop
avx2_k_tail:
	// Reduce Y1 into X0
	VEXTRACTF128 $1, Y1, X2
	VADDPD       X2, X1, X1
	VHADDPD      X1, X1, X1
	ADDSD        X1, X0
	// scalar tail
	CMPQ R11, $0
	JLE  avx2_k_done
avx2_scalar:
	MOVSD (R9), X1
	MOVSD (R10), X2
	MULSD X2, X1
	ADDSD X1, X0
	ADDQ $8, R9
	ADDQ $8, R10
	DECQ R11
	JNZ  avx2_scalar
avx2_k_done:
	MOVSD X0, fSUM(SP)
	VZEROUPPER
	JMP  next_ic

scalar_k_loop:
	// Scalar loop over k with boundary checks.
	MOVQ fKSIZE(SP), R8
	XORQ R9, R9           // k = 0
	MOVSD fSUM(SP), X0
scalar_k:
	CMPQ R9, R8
	JGE  scalar_k_done

	// inputIdx = lo*stride + k*dilation - pad
	MOVQ fLO(SP), R10
	MOVQ fSTRIDE(SP), R11
	IMULQ R11, R10
	MOVQ R9, R11
	MOVQ fDIL(SP), R12
	IMULQ R12, R11
	ADDQ R11, R10
	MOVQ fPAD(SP), R11
	SUBQ R11, R10         // R10 = inputIdx

	// bounds check: 0 <= inputIdx < l
	CMPQ R10, $0
	JL   skip_k
	MOVQ fL(SP), R11
	CMPQ R10, R11
	JGE  skip_k

	// x[inputBase + inputIdx]
	MOVQ BX, R11
	ADDQ R10, R11
	SHLQ $3, R11
	MOVQ fXP(SP), R12
	MOVSD (R12)(R11*1), X1

	// weight[ic*kSize + k] = weightRow[DX + k]
	MOVQ DX, R11
	ADDQ R9, R11
	SHLQ $3, R11
	MOVQ fWROWP(SP), R12
	MOVSD (R12)(R11*1), X2
	MULSD X2, X1
	ADDSD X1, X0
	JMP  next_k
skip_k:
next_k:
	INCQ R9
	JMP  scalar_k
scalar_k_done:
	MOVSD X0, fSUM(SP)

next_ic:
	MOVQ fIC(SP), AX
	INCQ AX
	MOVQ AX, fIC(SP)
	JMP  loop_ic

write_out:
	// out[ni*outC*lOut + oc*lOut + lo] = sum
	MOVQ fNI(SP), AX
	MOVQ fOUTC(SP), BX
	IMULQ BX, AX
	MOVQ fLOUT(SP), BX
	IMULQ BX, AX         // AX = ni*outC*lOut
	MOVQ fOC(SP), BX
	MOVQ fLOUT(SP), CX
	IMULQ CX, BX
	ADDQ BX, AX          // AX += oc*lOut
	ADDQ fLO(SP), AX     // AX += lo
	SHLQ $3, AX
	MOVQ fOUTP(SP), BX
	MOVSD fSUM(SP), X0
	MOVSD X0, (BX)(AX*1)

	MOVQ fLO(SP), AX
	INCQ AX
	MOVQ AX, fLO(SP)
	JMP  loop_lo

next_oc:
	MOVQ fOC(SP), AX
	INCQ AX
	MOVQ AX, fOC(SP)
	JMP  loop_oc

next_g:
	MOVQ fGRP(SP), AX
	INCQ AX
	MOVQ AX, fGRP(SP)
	JMP  loop_g

next_ni:
	MOVQ fNI(SP), AX
	INCQ AX
	MOVQ AX, fNI(SP)
	JMP  loop_ni

done_conv1d:
	VZEROUPPER
	RET
