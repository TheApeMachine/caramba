#include "textflag.h"

// RoPEAVX2(dst, src, cosTable, sinTable []float64, numPairs int)
// ABI0 offsets (each slice = ptr+len+cap = 24 bytes):
//   dst+0(FP)   ptr
//   src+24(FP)  ptr  src_len+32(FP)  len
//   cos+48(FP)  ptr
//   sin+72(FP)  ptr
//   numPairs+96(FP)  int
//
// Algorithm per pair i:
//   d[2i]   = src[2i]*cos[i] - src[2i+1]*sin[i]
//   d[2i+1] = src[2i]*sin[i] + src[2i+1]*cos[i]
//
// AVX2: process 4 pairs (8 float64s) per iteration.
// Layout in registers:
//   Y_x0   = [x0, x1, x2, x3, x4, x5, x6, x7]   (src, 4 pairs)
//   Y_cos  = [c0, c1, c2, c3]                      (4 cos values)
//   Y_sin  = [s0, s1, s2, s3]                      (4 sin values)
//
// Expand: broadcast each cos/sin pair to both lanes of the pair:
//   Y_cc = [c0,c0, c1,c1, c2,c2, c3,c3]
//   Y_ss = [s0,s0, s1,s1, s2,s2, s3,s3]
// VPERMILPD swaps within 128-bit lane: x1<->x0 etc.
//   Y_xp  = [x1, x0, x3, x2, x5, x4, x7, x6]
//
//   out_even = x0*c0 - x1*s0  -> fma: x_even*cos - x_odd*sin
//   out_odd  = x0*s0 + x1*c0  -> fma: x_odd*cos  + x_even*sin
//
// We split even/odd with VPERMILPD+blend:
//   Y_even = [x0, x0, x2, x2, x4, x4, x6, x6]  (broadcast even within pair)
//   Y_odd  = [x1, x1, x3, x3, x5, x5, x7, x7]
//
// Actually simpler:
//   d[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
//   d[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
//
// Load x as Y0 = [x0,x1, x2,x3, x4,x5, x6,x7]
// Expand cos: load 4 doubles Y_c=[c0,c1,c2,c3], then interleave:
//   Y_cc[0..7] = [c0,c0,c1,c1,c2,c2,c3,c3]  via VUNPCKLPD + VUNPCKHPD + VPERM2F128
// Similarly Y_ss.
// VPERMILPD 0x55 within 256 swaps pairs: Y_xswap = [x1,x0,x3,x2,x5,x4,x7,x6]
//
//   out = x*cc - xswap*ss  (for even output positions)
//     and x*ss + xswap*cc  (for odd output positions)
//
// Blend even/odd results:
//   VBLENDPD 0xAA selects odd lanes from second operand.

TEXT ·RoPEAVX2(SB), NOSPLIT, $0-104
	MOVQ dst+0(FP),      AX   // dst ptr
	MOVQ src+24(FP),     SI   // src ptr
	MOVQ cosTable+48(FP),     CX   // cos ptr
	MOVQ sinTable+72(FP),     DX   // sin ptr
	MOVQ numPairs+96(FP), BX  // numPairs

	// Process 4 pairs (8 float64s) per iteration
avx2loop:
	CMPQ BX, $4
	JL   avx2tail

	// Load x[0..7]
	VMOVUPD (SI), Y0

	// Load cos[0..3] and sin[0..3]
	VMOVUPD (CX), X1          // xmm1 = [c0, c1]
	VMOVUPD 16(CX), X2        // xmm2 = [c2, c3]
	VMOVUPD (DX), X3          // xmm3 = [s0, s1]
	VMOVUPD 16(DX), X4        // xmm4 = [s2, s3]

	// Interleave to get [c0,c0,c1,c1] in lower 128, [c2,c2,c3,c3] in upper 128
	VUNPCKLPD X1, X1, X5      // [c0,c0]
	VUNPCKHPD X1, X1, X6      // [c1,c1]
	VUNPCKLPD X2, X2, X7      // [c2,c2]
	VUNPCKHPD X2, X2, X8      // [c3,c3]
	VINSERTF128 $1, X6, Y5, Y5  // Y5 = [c0,c0, c1,c1]
	VINSERTF128 $1, X8, Y7, Y7  // Y7 = [c2,c2, c3,c3]
	VPERM2F128 $0x20, Y7, Y5, Y9 // Y9 = [c0,c0, c1,c1, c2,c2, c3,c3] -- cos broadcast

	VUNPCKLPD X3, X3, X5
	VUNPCKHPD X3, X3, X6
	VUNPCKLPD X4, X4, X7
	VUNPCKHPD X4, X4, X8
	VINSERTF128 $1, X6, Y5, Y5
	VINSERTF128 $1, X8, Y7, Y7
	VPERM2F128 $0x20, Y7, Y5, Y10 // Y10 = sin broadcast

	// Y_xswap: swap pairs within 128-bit lane: [x1,x0, x3,x2, x5,x4, x7,x6]
	VPERMILPD $0x05, Y0, Y11   // 0b00000101

	// even result: x*cos - xswap*sin
	VMULPD   Y9,  Y0,  Y12    // x*cos
	VMULPD   Y10, Y11, Y13    // xswap*sin
	VSUBPD   Y13, Y12, Y14    // x*cos - xswap*sin  -> even lanes correct

	// odd result: xswap*cos + x*sin  (but xswap has x_even in odd positions)
	// Actually: d[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
	//   odd lanes of Y0 are x[2i+1], odd lanes of Y11 are x[2i]
	VMULPD   Y10, Y0,  Y15    // x*sin
	VMULPD   Y9,  Y11, Y1     // xswap*cos
	VADDPD   Y1,  Y15, Y2     // x*sin + xswap*cos  -> odd lanes correct

	// Blend: take even lanes from Y14, odd lanes from Y2
	// VBLENDPD imm=0xAA = 1010 1010 selects from second src for odd lanes
	VBLENDPD $0xAA, Y2, Y14, Y3

	VMOVUPD Y3, (AX)

	ADDQ $64, SI
	ADDQ $64, AX
	ADDQ $32, CX
	ADDQ $32, DX
	SUBQ $4, BX
	JMP avx2loop

avx2tail:
	// Scalar fallback for remaining pairs
	CMPQ BX, $0
	JLE  done
scalarloop:
	MOVSD (SI),   X0   // x[2i]
	MOVSD 8(SI),  X1   // x[2i+1]
	MOVSD (CX),   X2   // cos[i]
	MOVSD (DX),   X3   // sin[i]

	// d[2i]   = x0*cos - x1*sin
	MOVAPD X0, X4
	MULSD  X2, X4      // x0*cos
	MOVAPD X1, X5
	MULSD  X3, X5      // x1*sin
	SUBSD  X5, X4
	MOVSD  X4, (AX)

	// d[2i+1] = x0*sin + x1*cos
	MULSD  X3, X0      // x0*sin
	MULSD  X2, X1      // x1*cos
	ADDSD  X1, X0
	MOVSD  X0, 8(AX)

	ADDQ $16, SI
	ADDQ $16, AX
	ADDQ $8,  CX
	ADDQ $8,  DX
	DECQ BX
	JNZ  scalarloop

done:
	VZEROUPPER
	RET
