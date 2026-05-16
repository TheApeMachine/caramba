#include "textflag.h"

#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFMINNM_D2(m, n, d) WORD $(0x4EE0C400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMAXNM_D2(m, n, d) WORD $(0x4E60C400 | ((m) << 16) | ((n) << 5) | (d))
#define VFRINTN_D2(n, d) WORD $(0x4E618800 | ((n) << 5) | (d))
#define VFCVTZS_D2(n, d) WORD $(0x4EE1B800 | ((n) << 5) | (d))
#define VLOADDUP(sym, addr, vec) MOVD $sym, addr; VLD1R (addr), [vec.D2]

// hawkesKernelRowNEON writes out[i] = alpha * exp(-beta * (events[i] - ti)).
TEXT ·hawkesKernelRowNEON(SB), NOSPLIT, $24-72
	MOVD out+0(FP), R0
	MOVD events+24(FP), R1
	MOVD out_len+8(FP), R2
	CBZ R2, hkr_done

	FMOVD ti+48(FP), F20
	FMOVD F20, 0(RSP)
	VLD1R (RSP), [V20.D2]
	FMOVD beta+64(FP), F21
	FMOVD F21, 8(RSP)
	ADD $8, RSP, R11
	VLD1R (R11), [V21.D2]
	FMOVD alpha+56(FP), F19
	FMOVD F19, 16(RSP)
	ADD $16, RSP, R11
	VLD1R (R11), [V19.D2]
	VLOADDUP(·hexLog2E(SB), R9, V22)
	VLOADDUP(·hexLn2Hi(SB), R9, V23)
	VLOADDUP(·hexLn2Lo(SB), R9, V24)
	VLOADDUP(·hexMaxArg(SB), R9, V25)
	VLOADDUP(·hexMinArg(SB), R9, V26)
	VLOADDUP(·hexC11(SB), R9, V5)
	VLOADDUP(·hexC10(SB), R9, V6)
	VLOADDUP(·hexC9(SB), R9, V7)
	VLOADDUP(·hexC8(SB), R9, V8)
	VLOADDUP(·hexC7(SB), R9, V9)
	VLOADDUP(·hexC6(SB), R9, V10)
	VLOADDUP(·hexC5(SB), R9, V11)
	VLOADDUP(·hexC4(SB), R9, V12)
	VLOADDUP(·hexC3(SB), R9, V13)
	VLOADDUP(·hexC2(SB), R9, V14)
	VLOADDUP(·hexC1(SB), R9, V15)
	VLOADDUP(·hexC0(SB), R9, V17)
	VEOR V4.B16, V4.B16, V4.B16
	MOVD $1023, R10
	VDUP R10, V27.D2
	LSR  $1, R2, R3
	CBZ  R3, hkr_tail

hkr_loop:
	VLD1.P 16(R1), [V0.D2]
	VFSUB_D2(20, 0, 0)
	VFMUL_D2(21, 0, 0)
	VFSUB_D2(0, 4, 0)
	VFMINNM_D2(25, 0, 0)
	VFMAXNM_D2(26, 0, 0)

	VFMUL_D2(22, 0, 1)
	VFRINTN_D2(1, 1)
	VFMUL_D2(23, 1, 3)
	VFSUB_D2(3, 0, 0)
	VFMUL_D2(24, 1, 3)
	VFSUB_D2(3, 0, 0)

	VORR V5.B16, V5.B16, V2.B16
	VFMUL_D2(0, 2, 2); VFADD_D2(6, 2, 2)
	VFMUL_D2(0, 2, 2); VFADD_D2(7, 2, 2)
	VFMUL_D2(0, 2, 2); VFADD_D2(8, 2, 2)
	VFMUL_D2(0, 2, 2); VFADD_D2(9, 2, 2)
	VFMUL_D2(0, 2, 2); VFADD_D2(10, 2, 2)
	VFMUL_D2(0, 2, 2); VFADD_D2(11, 2, 2)
	VFMUL_D2(0, 2, 2); VFADD_D2(12, 2, 2)
	VFMUL_D2(0, 2, 2); VFADD_D2(13, 2, 2)
	VFMUL_D2(0, 2, 2); VFADD_D2(14, 2, 2)
	VFMUL_D2(0, 2, 2); VFADD_D2(15, 2, 2)
	VFMUL_D2(0, 2, 2); VFADD_D2(17, 2, 2)

	VFCVTZS_D2(1, 1)
	VADD V27.D2, V1.D2, V1.D2
	VSHL $52, V1.D2, V1.D2
	VFMUL_D2(1, 2, 2)
	VFMUL_D2(19, 2, 2)
	VST1.P [V2.D2], 16(R0)

	SUBS $1, R3, R3
	BNE  hkr_loop

hkr_tail:
	TST $1, R2
	BEQ hkr_done

	FMOVD ti+48(FP), F20
	FMOVD beta+64(FP), F21
	FMOVD ·hexLog2E(SB), F22
	FMOVD ·hexLn2Hi(SB), F23
	FMOVD ·hexLn2Lo(SB), F24
	FMOVD ·hexMaxArg(SB), F25
	FMOVD ·hexMinArg(SB), F26
	FMOVD alpha+56(FP), F27
	FMOVD ·hexC11(SB), F5
	FMOVD ·hexC10(SB), F6
	FMOVD ·hexC9(SB), F7
	FMOVD ·hexC8(SB), F8
	FMOVD ·hexC7(SB), F9
	FMOVD ·hexC6(SB), F10
	FMOVD ·hexC5(SB), F11
	FMOVD ·hexC4(SB), F12
	FMOVD ·hexC3(SB), F13
	FMOVD ·hexC2(SB), F14
	FMOVD ·hexC1(SB), F15
	FMOVD ·hexC0(SB), F17

	FMOVD (R1), F0
	FSUBD F20, F0, F0                          // events - ti
	FMULD F21, F0, F0                          // *beta
	FNEGD F0, F0                               // -beta*(events-ti)
	FMINNMD F25, F0, F0
	FMAXNMD F26, F0, F0

	FMULD F22, F0, F1
	FRINTND F1, F1
	FMSUBD F1, F0, F23, F0
	FMSUBD F1, F0, F24, F0

	FMOVD  F5, F2                              // y = C11
	FMADDD F2, F6, F0, F2
	FMADDD F2, F7, F0, F2
	FMADDD F2, F8, F0, F2
	FMADDD F2, F9, F0, F2
	FMADDD F2, F10, F0, F2
	FMADDD F2, F11, F0, F2
	FMADDD F2, F12, F0, F2
	FMADDD F2, F13, F0, F2
	FMADDD F2, F14, F0, F2
	FMADDD F2, F15, F0, F2
	FMADDD F2, F17, F0, F2

	FCVTZSD F1, R5
	ADD R10, R5, R5
	LSL $52, R5, R5
	FMOVD R5, F4
	FMULD F4, F2, F2
	FMULD F27, F2, F2                          // *alpha
	FMOVD F2, (R0)

hkr_done:
	RET
