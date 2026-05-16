#include "textflag.h"

#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFMAXNM_D2(m, n, d) WORD $(0x4E60C400 | ((m) << 16) | ((n) << 5) | (d))

// LeakyReLUNEON(dst, src []float64, alpha float64)
// ABI0: dst+0(FP)=ptr, dst_len+8(FP)=len, dst_cap+16(FP)=cap,
//       src_base+24(FP)=ptr, src_len+32(FP)=len, src_cap+40(FP)=cap,
//       alpha+48(FP)=float64
//
// out[i] = max(alpha*x, x)

TEXT ·LeakyReLUNEON(SB),NOSPLIT,$8-56
	MOVD  dst+0(FP), R0
	MOVD  src_base+24(FP), R3
	MOVD  src_len+32(FP), R4
	FMOVD alpha+48(FP), F30
	FMOVD F30, 0(RSP)
	VLD1R (RSP), [V30.D2]

	LSR  $1, R4, R5
	CBZ  R5, tail

pairloop:
	VLD1.P 16(R3), [V1.D2]
	VFMUL_D2(30, 1, 2)
	VFMAXNM_D2(2, 1, 1)
	VST1.P [V1.D2], 16(R0)
	SUBS $1, R5, R5
	BNE  pairloop

tail:
	TST $1, R4
	BEQ done
	FMOVD (R3), F1
	FMULD F30, F1, F2
	FMAXD F2, F1, F1
	FMOVD F1, (R0)

done:
	RET
