// SPDX-License-Identifier: Apache-2.0
// NEON parameterized activation kernels (extra).
#include "textflag.h"

#define VFMUL_S4(m, n, d)  WORD $(0x6E20DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_S4(m, n, d)  WORD $(0x4EA0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFDIV_S4(m, n, d)  WORD $(0x6E20FC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFMLA_S4(m, n, d)  WORD $(0x4E20CC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFRINTN_S4(n, d)   WORD $(0x4E218800 | ((n) << 5) | (d))
#define VFCVTZS_S4(n, d)   WORD $(0x4EA1B800 | ((n) << 5) | (d))
#define VADD_S4(m, n, d)   WORD $(0x4EA08400 | ((m) << 16) | ((n) << 5) | (d))
#define VSHL_S4_BY23(n, d) WORD $(0x4F375400 | ((n) << 5) | (d))
#define VMOV_B16(src, dst) WORD $(0x4EA01C00 | ((src) << 16) | ((src) << 5) | (dst))
#define VFCMGT_S4(m, n, d) WORD $(0x6EA0E400 | ((m) << 16) | ((n) << 5) | (d))
#define VFCMLE_S4(m, n, d) WORD $(0x6EA0E000 | ((m) << 16) | ((n) << 5) | (d))
#define VBSL_B16(m, n, d)  WORD $(0x6E601C00 | ((m) << 16) | ((n) << 5) | (d))
#define VFABS_S4(n, d)     WORD $(0x6EA0F000 | ((n) << 5) | (d))
#define VFNEG_S4(n, d)     WORD $(0x6EA0F800 | ((n) << 5) | (d))
#define VFADD_S4(m, n, d)  WORD $(0x4E20D400 | ((m) << 16) | ((n) << 5) | (d))
#define VSCVTF_S4(n, d)    WORD $(0x4E21D800 | ((n) << 5) | (d))
#define VMUL_I32_S4(m, n, d) WORD $(0x4EA09C00 | ((m) << 16) | ((n) << 5) | (d))

#define NEON_EXP_BODY(in, out) \
    VFMUL_S4(16, in, 1) ;\
    VFRINTN_S4(1, 1) ;\
    VFMUL_S4(17, 1, 2) ;\
    VFSUB_S4(2, in, in) ;\
    VMOV_B16(19, 3) ;\
    VMOV_B16(20, 4) ; VFMLA_S4(in, 3, 4) ;\
    VMOV_B16(21, 3) ; VFMLA_S4(in, 4, 3) ;\
    VMOV_B16(22, 4) ; VFMLA_S4(in, 3, 4) ;\
    VMOV_B16(23, 3) ; VFMLA_S4(in, 4, 3) ;\
    VMOV_B16(24, 4) ; VFMLA_S4(in, 3, 4) ;\
    VMOV_B16(25, 3) ; VFMLA_S4(in, 4, 3) ;\
    VMOV_B16(26, 4) ; VFMLA_S4(in, 3, 4) ;\
    VFCVTZS_S4(1, 5) ;\
    VADD_S4(27, 5, 5) ;\
    VSHL_S4_BY23(5, 5) ;\
    VFMUL_S4(5, 4, out)

DATA actParamSnakeC<>+0(SB)/4, $6.283185307179586
DATA actParamSnakeC<>+4(SB)/4, $3.141592653589793
DATA actParamSnakeC<>+8(SB)/4, $0.16666667
DATA actParamSnakeC<>+12(SB)/4, $0.008333333
DATA actParamSnakeC<>+16(SB)/4, $5.9604645e-08
GLOBL actParamSnakeC<>(SB), 8, $20

DATA actExtraExpC<>+0(SB)/4, $1.4426950408889634
DATA actExtraExpC<>+4(SB)/4, $0.6931471805599453
DATA actExtraExpC<>+8(SB)/4, $127.0
DATA actExtraExpC<>+12(SB)/4, $0.00019841270
DATA actExtraExpC<>+16(SB)/4, $0.0013888889
DATA actExtraExpC<>+20(SB)/4, $0.008333334
DATA actExtraExpC<>+24(SB)/4, $0.041666667
DATA actExtraExpC<>+28(SB)/4, $0.16666667
DATA actExtraExpC<>+32(SB)/4, $0.5
DATA actExtraExpC<>+36(SB)/4, $1.0
DATA actExtraExpC<>+40(SB)/4, $1.0
DATA actExtraExpC<>+44(SB)/4, $2.0
GLOBL actExtraExpC<>(SB), 8, $48

// func ELUAlphaF32NEON(dst, src *float32, count int, alpha float32)
TEXT ·ELUAlphaF32NEON(SB), NOSPLIT, $0-28
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD count+16(FP), R2
	FMOVS alpha+24(FP), F30
	VDUP V30.S[0], V30.S4
	MOVD $actExtraExpC(SB), R3
	FMOVS  0(R3), F16
	FMOVS  4(R3), F17
	FMOVS  8(R3), F18
	FMOVS 12(R3), F19
	FMOVS 16(R3), F20
	FMOVS 20(R3), F21
	FMOVS 24(R3), F22
	FMOVS 28(R3), F23
	FMOVS 32(R3), F24
	FMOVS 36(R3), F25
	FMOVS 40(R3), F26
	VFCVTZS_S4(18, 27)
	FMOVS 44(R3), F28
	VEOR V8.B16, V8.B16, V8.B16
ela_neon_w4:
	CMP $4, R2
	BLT ela_neon_scalar
	VLD1.P 16(R1), [V0.S4]
	VMOV_B16(0, 10)
	VFCMGT_S4(8, 0, 4)
	NEON_EXP_BODY(0, 6)
	VFSUB_S4(26, 6, 6)
	VFMUL_S4(30, 6, 6)
	VBSL_B16(4, 10, 7)
	VST1.P [V7.S4], 16(R0)
	SUB $4, R2
	B ela_neon_w4
ela_neon_scalar:
	CBZ R2, ela_neon_done
ela_neon_sloop:
	FMOVS (R1), F0
	VDUP V0.S[0], V0.S4
	VMOV_B16(0, 10)
	VFCMGT_S4(8, 0, 4)
	NEON_EXP_BODY(0, 6)
	VFSUB_S4(26, 6, 6)
	VFMUL_S4(30, 6, 6)
	VBSL_B16(4, 10, 7)
	FMOVS F7, (R0)
	ADD $4, R1
	ADD $4, R0
	SUB $1, R2
	CBNZ R2, ela_neon_sloop
ela_neon_done:
	RET

// func CELUAlphaF32NEON(dst, src *float32, count int, alpha float32)
TEXT ·CELUAlphaF32NEON(SB), NOSPLIT, $0-28
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD count+16(FP), R2
	FMOVS alpha+24(FP), F30
	VDUP V30.S[0], V30.S4
	MOVD $actExtraExpC(SB), R3
	FMOVS  0(R3), F16
	FMOVS  4(R3), F17
	FMOVS  8(R3), F18
	FMOVS 12(R3), F19
	FMOVS 16(R3), F20
	FMOVS 20(R3), F21
	FMOVS 24(R3), F22
	FMOVS 28(R3), F23
	FMOVS 32(R3), F24
	FMOVS 36(R3), F25
	FMOVS 40(R3), F26
	VFCVTZS_S4(18, 27)
	FMOVS 44(R3), F28
	VEOR V8.B16, V8.B16, V8.B16
cla_neon_w4:
	CMP $4, R2
	BLT cla_neon_scalar
	VLD1.P 16(R1), [V0.S4]
	VMOV_B16(0, 10)
	VFCMGT_S4(8, 0, 4)
	VMOV_B16(0, 9)
	VFDIV_S4(30, 0, 9)
	VBSL_B16(4, 0, 9)
	NEON_EXP_BODY(9, 6)
	VFSUB_S4(26, 6, 6)
	VFMUL_S4(30, 6, 6)
	VBSL_B16(4, 10, 7)
	VST1.P [V7.S4], 16(R0)
	SUB $4, R2
	B cla_neon_w4
cla_neon_scalar:
	CBZ R2, cla_neon_done
cla_neon_sloop:
	FMOVS (R1), F0
	VDUP V0.S[0], V0.S4
	VMOV_B16(0, 10)
	VFCMGT_S4(8, 0, 4)
	VMOV_B16(0, 9)
	VFDIV_S4(30, 0, 9)
	VBSL_B16(4, 0, 9)
	NEON_EXP_BODY(9, 6)
	VFSUB_S4(26, 6, 6)
	VFMUL_S4(30, 6, 6)
	VBSL_B16(4, 10, 7)
	FMOVS F7, (R0)
	ADD $4, R1
	ADD $4, R0
	SUB $1, R2
	CBNZ R2, cla_neon_sloop
cla_neon_done:
	RET

// func HardShrinkF32NEON(dst, src *float32, count int, lambda float32)
TEXT ·HardShrinkF32NEON(SB), NOSPLIT, $0-28
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD count+16(FP), R2
	FMOVS lambda+24(FP), F29
	VDUP V29.S[0], V29.S4
	VEOR V8.B16, V8.B16, V8.B16
hs_neon_w4:
	CMP $4, R2
	BLT hs_neon_scalar
	VLD1.P 16(R1), [V0.S4]
	VFABS_S4(0, 1)
	VFCMGT_S4(29, 1, 4)
	VBSL_B16(4, 0, 7)
	VST1.P [V7.S4], 16(R0)
	SUB $4, R2
	B hs_neon_w4
hs_neon_scalar:
	CBZ R2, hs_neon_done
hs_neon_sloop:
	FMOVS (R1), F0
	VDUP V0.S[0], V0.S4
	VFABS_S4(0, 1)
	VFCMGT_S4(29, 1, 4)
	VBSL_B16(4, 0, 7)
	FMOVS F7, (R0)
	ADD $4, R1
	ADD $4, R0
	SUB $1, R2
	CBNZ R2, hs_neon_sloop
hs_neon_done:
	RET

// func SoftShrinkF32NEON(dst, src *float32, count int, lambda float32)
TEXT ·SoftShrinkF32NEON(SB), NOSPLIT, $0-28
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD count+16(FP), R2
	FMOVS lambda+24(FP), F10
	VDUP V10.S[0], V10.S4
	VFNEG_S4(10, 11)
	VEOR V12.B16, V12.B16, V12.B16
ss_neon_w4:
	CMP $4, R2
	BLT ss_neon_scalar
	VLD1.P 16(R1), [V0.S4]
	VFCMGT_S4(10, 0, 1)
	VFCMGT_S4(0, 11, 2)
	VFSUB_S4(10, 0, 3)
	VFADD_S4(10, 0, 4)
	VBSL_B16(12, 3, 1)
	VBSL_B16(1, 4, 2)
	VST1.P [V2.S4], 16(R0)
	SUB $4, R2
	B ss_neon_w4
ss_neon_scalar:
	CBZ R2, ss_neon_done
ss_neon_sloop:
	FMOVS (R1), F0
	VDUP V0.S[0], V0.S4
	VFCMGT_S4(10, 0, 1)
	VFCMGT_S4(0, 11, 2)
	VFSUB_S4(10, 0, 3)
	VFADD_S4(10, 0, 4)
	VBSL_B16(12, 3, 1)
	VBSL_B16(1, 4, 2)
	FMOVS F2, (R0)
	ADD $4, R1
	ADD $4, R0
	SUB $1, R2
	CBNZ R2, ss_neon_sloop
ss_neon_done:
	RET

// func SnakeF32NEON(dst, src *float32, count int, alpha float32)
TEXT ·SnakeF32NEON(SB), NOSPLIT, $0-28
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD count+16(FP), R2
	FMOVS alpha+24(FP), F10
	VDUP V10.S[0], V10.S4
	FMOVS $1.0, F18
	VDUP V18.S[0], V18.S4
	VFDIV_S4(10, 18, 19)
	MOVD $actParamSnakeC<>(SB), R3
	FMOVS 0(R3), F11
	VDUP V11.S[0], V11.S4
	FMOVS 4(R3), F12
	VDUP V12.S[0], V12.S4
	FMOVS 8(R3), F13
	VDUP V13.S[0], V13.S4
	FMOVS 12(R3), F14
	VDUP V14.S[0], V14.S4
	VFNEG_S4(11, 15)
	VFNEG_S4(12, 16)
	VEOR V17.B16, V17.B16, V17.B16
snake_neon_w4:
	CMP $4, R2
	BLT snake_neon_scalar
	VLD1.P 16(R1), [V0.S4]
	VFMUL_S4(10, 0, 1)
	VFDIV_S4(11, 1, 2)
	VFRINTN_S4(2, 2)
	VFMUL_S4(11, 2, 2)
	VFSUB_S4(2, 1, 1)
	VFCMGT_S4(12, 1, 3)
	VBSL_B16(17, 15, 3)
	VFADD_S4(3, 1, 1)
	VFCMGT_S4(1, 16, 4)
	VBSL_B16(17, 11, 4)
	VFADD_S4(4, 1, 1)
	VFMUL_S4(1, 1, 2)
	VFMUL_S4(14, 2, 3)
	VFSUB_S4(3, 13, 3)
	VFMUL_S4(3, 2, 3)
	VFSUB_S4(3, 18, 3)
	VFMUL_S4(3, 1, 3)
	VFMUL_S4(3, 3, 3)
	VFMUL_S4(19, 3, 3)
	VFADD_S4(3, 0, 0)
	VST1.P [V0.S4], 16(R0)
	SUB $4, R2
	B snake_neon_w4
snake_neon_scalar:
	CBZ R2, snake_neon_done
snake_neon_sloop:
	FMOVS (R1), F0
	VDUP V0.S[0], V0.S4
	VFMUL_S4(10, 0, 1)
	VFDIV_S4(11, 1, 2)
	VFRINTN_S4(2, 2)
	VFMUL_S4(11, 2, 2)
	VFSUB_S4(2, 1, 1)
	VFCMGT_S4(12, 1, 3)
	VBSL_B16(17, 15, 3)
	VFADD_S4(3, 1, 1)
	VFCMGT_S4(1, 16, 4)
	VBSL_B16(17, 11, 4)
	VFADD_S4(4, 1, 1)
	VFMUL_S4(1, 1, 2)
	VFMUL_S4(14, 2, 3)
	VFSUB_S4(3, 13, 3)
	VFMUL_S4(3, 2, 3)
	VFSUB_S4(3, 18, 3)
	VFMUL_S4(3, 1, 3)
	VFMUL_S4(3, 3, 3)
	VFMUL_S4(19, 3, 3)
	VFADD_S4(3, 0, 0)
	FMOVS F0, (R0)
	ADD $4, R1
	ADD $4, R0
	SUB $1, R2
	CBNZ R2, snake_neon_sloop
snake_neon_done:
	RET

// func SnakeParametricF32NEON(dst, src *float32, count int, alpha, beta float32)
TEXT ·SnakeParametricF32NEON(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD count+16(FP), R2
	FMOVS alpha+24(FP), F10
	VDUP V10.S[0], V10.S4
	FMOVS beta+28(FP), F20
	VDUP V20.S[0], V20.S4
	FMOVS $1.0, F18
	VDUP V18.S[0], V18.S4
	VFDIV_S4(20, 18, 19)
	MOVD $actParamSnakeC<>(SB), R3
	FMOVS 0(R3), F11
	VDUP V11.S[0], V11.S4
	FMOVS 4(R3), F12
	VDUP V12.S[0], V12.S4
	FMOVS 8(R3), F13
	VDUP V13.S[0], V13.S4
	FMOVS 12(R3), F14
	VDUP V14.S[0], V14.S4
	VFNEG_S4(11, 15)
	VFNEG_S4(12, 16)
	VEOR V17.B16, V17.B16, V17.B16
snakep_neon_w4:
	CMP $4, R2
	BLT snakep_neon_scalar
	VLD1.P 16(R1), [V0.S4]
	VFMUL_S4(10, 0, 1)
	VFDIV_S4(11, 1, 2)
	VFRINTN_S4(2, 2)
	VFMUL_S4(11, 2, 2)
	VFSUB_S4(2, 1, 1)
	VFCMGT_S4(12, 1, 3)
	VBSL_B16(17, 15, 3)
	VFADD_S4(3, 1, 1)
	VFCMGT_S4(1, 16, 4)
	VBSL_B16(17, 11, 4)
	VFADD_S4(4, 1, 1)
	VFMUL_S4(1, 1, 2)
	VFMUL_S4(14, 2, 3)
	VFSUB_S4(3, 13, 3)
	VFMUL_S4(3, 2, 3)
	VFSUB_S4(3, 18, 3)
	VFMUL_S4(3, 1, 3)
	VFMUL_S4(3, 3, 3)
	VFMUL_S4(19, 3, 3)
	VFADD_S4(3, 0, 0)
	VST1.P [V0.S4], 16(R0)
	SUB $4, R2
	B snakep_neon_w4
snakep_neon_scalar:
	CBZ R2, snakep_neon_done
snakep_neon_sloop:
	FMOVS (R1), F0
	VDUP V0.S[0], V0.S4
	VFMUL_S4(10, 0, 1)
	VFDIV_S4(11, 1, 2)
	VFRINTN_S4(2, 2)
	VFMUL_S4(11, 2, 2)
	VFSUB_S4(2, 1, 1)
	VFCMGT_S4(12, 1, 3)
	VBSL_B16(17, 15, 3)
	VFADD_S4(3, 1, 1)
	VFCMGT_S4(1, 16, 4)
	VBSL_B16(17, 11, 4)
	VFADD_S4(4, 1, 1)
	VFMUL_S4(1, 1, 2)
	VFMUL_S4(14, 2, 3)
	VFSUB_S4(3, 13, 3)
	VFMUL_S4(3, 2, 3)
	VFSUB_S4(3, 18, 3)
	VFMUL_S4(3, 1, 3)
	VFMUL_S4(3, 3, 3)
	VFMUL_S4(19, 3, 3)
	VFADD_S4(3, 0, 0)
	FMOVS F0, (R0)
	ADD $4, R1
	ADD $4, R0
	SUB $1, R2
	CBNZ R2, snakep_neon_sloop
snakep_neon_done:
	RET

// func RReLUF32NEON(dst, src *float32, count int, lower, upper float32)
TEXT ·RReLUF32NEON(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD count+16(FP), R2
	MOVD $0xA5A5A5A5, R8
	MOVD lower+24(FP), R9
	EOR R9, R8
	MOVD upper+28(FP), R9
	EOR R9, R8
	FMOVS lower+24(FP), F10
	VDUP V10.S[0], V10.S4
	FMOVS upper+28(FP), F11
	VDUP V11.S[0], V11.S4
	VFSUB_S4(10, 11, 12)
	FMOVS actParamSnakeC<>+16(SB), F13
	VDUP V13.S[0], V13.S4
	VEOR V14.B16, V14.B16, V14.B16
	MOVD $0x00FFFFFF, R10
	VDUP R10, V6.S4
	MOVD $1664525, R11
	MOVD $1013904223, R12
	VMOV R8, V2.S[0]
	MUL R11, R8
	ADD R12, R8
	VMOV R8, V2.S[1]
	MUL R11, R8
	ADD R12, R8
	VMOV R8, V2.S[2]
	MUL R11, R8
	ADD R12, R8
	VMOV R8, V2.S[3]
	MUL R11, R8
	ADD R12, R8
	MOVD $158984081, R11
	VDUP R11, V7.S4
	MOVD $2868466484, R12
	VDUP R12, V8.S4
rr_neon_w4:
	CMP $4, R2
	BLT rr_neon_scalar
	VLD1.P 16(R1), [V0.S4]
	VUSHR $8, V2.S4, V5.S4
	VAND V6.B16, V5.B16, V5.B16
	VSCVTF_S4(5, 5)
	VFMUL_S4(13, 5, 5)
	VFMUL_S4(12, 5, 5)
	VFADD_S4(10, 5, 5)
	VFMUL_S4(0, 5, 5)
	VFCMGT_S4(14, 0, 4)
	VBSL_B16(5, 0, 4)
	VST1.P [V4.S4], 16(R0)
	VMUL_I32_S4(7, 2, 2)
	VADD_S4(8, 2, 2)
	SUB $4, R2
	B rr_neon_w4
rr_neon_scalar:
	CBZ R2, rr_neon_done
	MOVD $1664525, R11
	MOVD $1013904223, R12
rr_neon_sloop:
	FMOVS (R1), F0
	FCMPS F14, F0
	BGT rr_neon_pos
	MOVD R8, R9
	MUL R11, R8
	ADD R12, R8
	LSR $8, R9
	AND $0x00FFFFFF, R9
	SCVTFS R9, F2
	FMULS F13, F2
	FMULS F12, F2
	FADDS F10, F2
	FMULS F2, F0
	FMOVS F0, (R0)
	B rr_neon_step
rr_neon_pos:
	FMOVS F0, (R0)
rr_neon_step:
	ADD $4, R1
	ADD $4, R0
	SUB $1, R2
	CBNZ R2, rr_neon_sloop
rr_neon_done:
	RET
