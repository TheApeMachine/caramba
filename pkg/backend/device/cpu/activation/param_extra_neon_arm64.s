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

// func ELUAlphaF32NEON(dst, src *float32, count int, alpha float32)
TEXT ·ELUAlphaF32NEON(SB), NOSPLIT, $0-28
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD count+16(FP), R2
	FMOVS alpha+24(FP), F30
	VDUP V30.S[0], V30.S4
	MOVD $actExtraExpC<>(SB), R3
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
	MOVD $actExtraExpC<>(SB), R3
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
	MOVD $0, R10
	FMOVS R10, F12
	FSUBS F10, F12, F11
ss_neon_loop:
	CBZ R2, ss_neon_done
	FMOVS (R1), F0
	FCMPS F0, F10
	BGT ss_neon_hi
	FCMPS F0, F11
	BLT ss_neon_lo
	FMOVS F12, (R0)
	B ss_neon_step
ss_neon_hi:
	FSUBS F10, F0, F0
	FMOVS F0, (R0)
	B ss_neon_step
ss_neon_lo:
	FADDS F11, F0, F0
	FMOVS F0, (R0)
ss_neon_step:
	ADD $4, R1
	ADD $4, R0
	SUB $1, R2
	B ss_neon_loop
ss_neon_done:
	RET

// func SnakeF32NEON(dst, src *float32, count int, alpha float32)
TEXT ·SnakeF32NEON(SB), NOSPLIT, $0-28
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD count+16(FP), R2
	FMOVS alpha+24(FP), F10
	MOVD $actParamSnakeC<>(SB), R3
snake_neon_loop:
	CBZ R2, snake_neon_done
	FMOVS (R1), F7
	FMOVS F7, F0
	FMULS F10, F0
	FMOVS F0, F14
	FMOVS (R3), F8
	FDIVS F8, F0, F0
	FRINTNS F0, F0
	FMOVS F0, F1
	FMOVS (R3), F8
	FMULS F8, F1
	FMOVS F14, F0
	FSUBS F1, F0
	FMOVS 4(R3), F9
	FCMPS F0, F9
	BLS snake_fold_lo
	FSUBS F8, F0
	B snake_sin
snake_fold_lo:
	FCMPS F9, F0
	BPL snake_sin
	FADDS F8, F0
snake_sin:
	FMOVS F0, F1
	FMULS F1, F1
	FMOVS 12(R3), F12
	FMULS F12, F1
	FMOVS 8(R3), F11
	FSUBS F1, F11
	FMOVS F0, F1
	FMULS F1, F1
	FMULS F11, F1
	FMOVS 8(R3), F11
	FSUBS F1, F11
	FMULS F0, F11
	FMOVS F11, F6
	FMULS F6, F6
	FDIVS F10, F6
	FADDS F7, F6
	FMOVS F6, (R0)
	ADD $4, R1
	ADD $4, R0
	SUB $1, R2
	B snake_neon_loop
snake_neon_done:
	RET

// func SnakeParametricF32NEON(dst, src *float32, count int, alpha, beta float32)
TEXT ·SnakeParametricF32NEON(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD count+16(FP), R2
	FMOVS alpha+24(FP), F10
	FMOVS beta+28(FP), F13
	MOVD $actParamSnakeC<>(SB), R3
snakep_neon_loop:
	CBZ R2, snakep_neon_done
	FMOVS (R1), F7
	FMOVS F7, F0
	FMULS F10, F0
	FMOVS F0, F14
	FMOVS (R3), F8
	FDIVS F8, F0, F0
	FRINTNS F0, F0
	FMOVS F0, F1
	FMOVS (R3), F8
	FMULS F8, F1
	FMOVS F14, F0
	FSUBS F1, F0
	FMOVS 4(R3), F9
	FCMPS F0, F9
	BLS snakep_fold_lo
	FSUBS F8, F0
	B snakep_sin
snakep_fold_lo:
	FCMPS F9, F0
	BPL snakep_sin
	FADDS F8, F0
snakep_sin:
	FMOVS F0, F1
	FMULS F1, F1
	FMOVS 12(R3), F12
	FMULS F12, F1
	FMOVS 8(R3), F11
	FSUBS F1, F11
	FMOVS F0, F1
	FMULS F1, F1
	FMULS F11, F1
	FMOVS 8(R3), F11
	FSUBS F1, F11
	FMULS F0, F11
	FMOVS F11, F6
	FMULS F6, F6
	FDIVS F13, F6
	FADDS F7, F6
	FMOVS F6, (R0)
	ADD $4, R1
	ADD $4, R0
	SUB $1, R2
	B snakep_neon_loop
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
	FMOVS upper+28(FP), F11
	FSUBS F10, F11, F12
	FMOVS actParamSnakeC<>+16(SB), F13
rr_neon_loop:
	CBZ R2, rr_neon_done
	FMOVS (R1), F0
	MOVD $0, R10
	FMOVS R10, F14
	FCMPS F0, F14
	BGT rr_neon_pos
	MOVD R8, R9
	MOVD $1664525, R10
	MUL R10, R9
	ADD $1013904223, R9
	MOVD R9, R8
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
	B rr_neon_loop
rr_neon_done:
	RET
