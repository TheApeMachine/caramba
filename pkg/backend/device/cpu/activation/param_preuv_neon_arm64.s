// SPDX-License-Identifier: Apache-2.0
// NEON PReLU with per-element slope vectors (count == slopeCount).
#include "textflag.h"

#define VFMUL_S4(m, n, d)  WORD $(0x6E20DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFCMGT_S4(m, n, d) WORD $(0x6EA0E400 | ((m) << 16) | ((n) << 5) | (d))
#define VBSL_B16(m, n, d)  WORD $(0x6E601C00 | ((m) << 16) | ((n) << 5) | (d))

// func PReLUVF32NEON(dst, src, slopes *float32, count int)
TEXT ·PReLUVF32NEON(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD slopes+16(FP), R2
	MOVD count+24(FP), R3
	VEOR V8.B16, V8.B16, V8.B16
preuv_neon_w4:
	CMP $4, R3
	BLT preuv_neon_scalar
	VLD1.P 16(R1), [V0.S4]
	VLD1.P 16(R2), [V10.S4]
	VFCMGT_S4(8, 0, 4)
	VFMUL_S4(10, 0, 5)
	VBSL_B16(4, 0, 7)
	VST1.P [V7.S4], 16(R0)
	SUB $4, R3
	B preuv_neon_w4
preuv_neon_scalar:
	CBZ R3, preuv_neon_done
preuv_neon_sloop:
	FMOVS (R1), F0
	FMOVS (R2), F10
	VDUP V0.S[0], V0.S4
	VDUP V10.S[0], V10.S4
	VFCMGT_S4(8, 0, 4)
	VFMUL_S4(10, 0, 5)
	VBSL_B16(4, 0, 7)
	FMOVS F7, (R0)
	ADD $4, R1
	ADD $4, R2
	ADD $4, R0
	SUB $1, R3
	CBNZ R3, preuv_neon_sloop
preuv_neon_done:
	RET
