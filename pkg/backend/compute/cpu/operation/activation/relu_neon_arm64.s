#include "textflag.h"

#define VFMAXNM_D2(m, n, d) WORD $(0x4E60C400 | ((m) << 16) | ((n) << 5) | (d))

DATA ·reluZero+0(SB)/8, $0.0
GLOBL ·reluZero(SB), RODATA|NOPTR, $8

// ReLUNEON(dst, src []float64)
// ABI0: dst+0(FP)=ptr, dst_len+8(FP)=len, dst_cap+16(FP)=cap,
//       src_base+24(FP)=ptr, src_len+32(FP)=len, src_cap+40(FP)=cap

TEXT ·ReLUNEON(SB),NOSPLIT,$0-48
	MOVD  dst+0(FP), R0
	MOVD  src_base+24(FP), R3
	MOVD  src_len+32(FP), R4
	FMOVD ·reluZero(SB), F31
	VEOR  V31.B16, V31.B16, V31.B16

	LSR  $1, R4, R5
	CBZ  R5, tail

pairloop:
	VLD1.P 16(R3), [V0.D2]
	VFMAXNM_D2(31, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS $1, R5, R5
	BNE  pairloop

tail:
	TST $1, R4
	BEQ done
	FMOVD (R3), F0
	FMAXD F31, F0, F0
	FMOVD F0, (R0)

done:
	RET
