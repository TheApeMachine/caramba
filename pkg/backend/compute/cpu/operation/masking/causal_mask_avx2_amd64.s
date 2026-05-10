#include "textflag.h"

// CausalMaskAVX2(dst []float64, seqLen int)
// ABI0: dst+0(FP)=ptr, dst_len+8(FP)=len, dst_cap+16(FP)=cap, seqLen+24(FP)=int
//
// For each row i (0..seqLen-1):
//   columns 0..i      => 0.0
//   columns i+1..seqLen-1 => -Inf
//
// Strategy: broadcast -Inf into Y15, use scalar zero stores for attended
// positions and VMOVUPD for masked portions (4 doubles at a time).

DATA ·avx2NegInf+0(SB)/8, $0xFFF0000000000000  // -Inf as float64
GLOBL ·avx2NegInf(SB), RODATA|NOPTR, $8

TEXT ·CausalMaskAVX2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), AX      // AX = dst base ptr
	MOVQ seqLen+24(FP), CX  // CX = seqLen (N)
	CMPQ CX, $0
	JLE  cm_avx2_done

	// Load -Inf scalar into X15, broadcast to Y15
	VBROADCASTSD ·avx2NegInf(SB), Y15
	VXORPD Y14, Y14, Y14    // Y14 = 0.0

	XORQ DX, DX             // DX = row i = 0

cm_avx2_row:
	CMPQ DX, CX
	JGE  cm_avx2_done

	// Compute pointer to start of row: AX already points there
	// Write i+1 zeros (columns 0..i)
	MOVQ DX, BX             // BX = i
	INCQ BX                 // BX = i+1 (number of zeros)
	MOVQ AX, DI             // DI = current write ptr

	// Write zeros scalar (no SIMD needed for small attended prefix)
cm_avx2_zero_loop:
	CMPQ BX, $0
	JLE  cm_avx2_zero_done
	MOVSD X14, (DI)
	ADDQ  $8, DI
	DECQ  BX
	JMP   cm_avx2_zero_loop

cm_avx2_zero_done:
	// Now write seqLen-i-1 -Inf values (columns i+1..seqLen-1)
	MOVQ CX, BX
	SUBQ DX, BX
	DECQ BX                 // BX = seqLen - i - 1

	// AVX2: 4 doubles per iteration
cm_avx2_inf_loop:
	CMPQ BX, $4
	JL   cm_avx2_inf_tail
	VMOVUPD Y15, (DI)
	ADDQ $32, DI
	SUBQ $4, BX
	JMP  cm_avx2_inf_loop

cm_avx2_inf_tail:
	CMPQ BX, $0
	JLE  cm_avx2_row_done
	VMOVSD X15, (DI)
	ADDQ $8, DI
	DECQ BX
	JMP  cm_avx2_inf_tail

cm_avx2_row_done:
	// Advance AX by seqLen * 8 bytes
	MOVQ CX, SI
	SHLQ $3, SI
	ADDQ SI, AX
	INCQ DX
	JMP  cm_avx2_row

cm_avx2_done:
	VZEROUPPER
	RET
