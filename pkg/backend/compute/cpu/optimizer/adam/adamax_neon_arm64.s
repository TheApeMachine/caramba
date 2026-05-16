#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFDIV_D2(m, n, d) WORD $(0x6E60FC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFMAXNM_D2(m, n, d) WORD $(0x4E60C400 | ((m) << 16) | ((n) << 5) | (d))

DATA ·adamaxAbsMask+0(SB)/8, $0x7FFFFFFFFFFFFFFF
GLOBL ·adamaxAbsMask(SB), RODATA, $8

// adamaxStepNEON(out, m, u, params, grads []float64,
//                beta1, oneMinusBeta1, beta2, lrT, eps float64)
TEXT ·adamaxStepNEON(SB), NOSPLIT, $48-160
	MOVD out+0(FP), R0
	MOVD m+24(FP), R1
	MOVD u+48(FP), R2
	MOVD params+72(FP), R3
	MOVD grads+96(FP), R4
	MOVD out_len+8(FP), R5

	FMOVD beta1+120(FP), F20
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	FMOVD oneMinusBeta1+128(FP), F21
	FMOVD F21, 8(RSP)
	ADD $8, RSP, R7
	VLD1R (R7), [V21.D2]
	FMOVD beta2+136(FP), F22
	FMOVD F22, 16(RSP)
	ADD $16, RSP, R7
	VLD1R (R7), [V22.D2]
	FMOVD lrT+144(FP), F23
	FMOVD F23, 24(RSP)
	ADD $24, RSP, R7
	VLD1R (R7), [V23.D2]
	FMOVD eps+152(FP), F24
	FMOVD F24, 32(RSP)
	ADD $32, RSP, R7
	VLD1R (R7), [V24.D2]
	MOVD ·adamaxAbsMask(SB), R10
	FMOVD R10, F25
	FMOVD F25, 40(RSP)
	ADD $40, RSP, R7
	VLD1R (R7), [V25.D2]

	LSR $1, R5, R6
	CBZ R6, adamax_neon_tail
adamax_neon_loop:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R2), [V2.D2]
	VLD1.P 16(R3), [V4.D2]
	VLD1.P 16(R4), [V6.D2]

	VFMUL_D2(20, 0, 0)
	VFMUL_D2(21, 6, 8)
	VFADD_D2(8, 0, 0)
	SUB $16, R1, R1
	VST1.P [V0.D2], 16(R1)

	VAND V25.B16, V6.B16, V8.B16
	VFMUL_D2(22, 2, 2)
	VFMAXNM_D2(8, 2, 2)
	SUB $16, R2, R2
	VST1.P [V2.D2], 16(R2)

	VFADD_D2(24, 2, 10)
	VFDIV_D2(10, 0, 12)
	VFMUL_D2(23, 12, 12)
	VFSUB_D2(12, 4, 4)
	VST1.P [V4.D2], 16(R0)

	SUBS $1, R6, R6
	BNE  adamax_neon_loop

adamax_neon_tail:
	AND $1, R5, R7
	CBZ R7, adamax_neon_done

	FMOVD (R1), F0
	FMOVD (R2), F2
	FMOVD (R3), F4
	FMOVD (R4), F6

	FMULD F20, F0, F0
	FMADDD F6, F0, F21, F0
	FMOVD F0, (R1)

	FMOVD F6, R11
	AND R10, R11, R11
	FMOVD R11, F8

	FMULD F22, F2, F2
	FMAXD F8, F2, F2
	FMOVD F2, (R2)

	FADDD F24, F2, F10
	FDIVD F10, F0, F12
	FMSUBD F23, F4, F12, F4
	FMOVD F4, (R0)

adamax_neon_done:
	RET
