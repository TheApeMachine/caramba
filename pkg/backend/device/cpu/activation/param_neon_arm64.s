// SPDX-License-Identifier: Apache-2.0
// NEON parameterized activation kernels.
#include "textflag.h"

#define VFMUL_S4(m, n, d)  WORD $(0x6E20DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFCMGT_S4(m, n, d) WORD $(0x6EA0E400 | ((m) << 16) | ((n) << 5) | (d))
#define VBSL_B16(m, n, d)  WORD $(0x6E601C00 | ((m) << 16) | ((n) << 5) | (d))
#define VMAX_S4(m, n, d)   WORD $(0x6EA0F400 | ((m) << 16) | ((n) << 5) | (d))
#define VMIN_S4(m, n, d)   WORD $(0x6EA0D400 | ((m) << 16) | ((n) << 5) | (d))

// func LeakyReLUSlopeF32NEON(dst, src *float32, count int, negativeSlope float32)
TEXT ·LeakyReLUSlopeF32NEON(SB), NOSPLIT, $0-28
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD count+16(FP), R2
    FMOVS negativeSlope+24(FP), F29
    VDUP V29.S[0], V29.S4
    VEOR V8.B16, V8.B16, V8.B16

lrs_neon_w4:
    CMP  $4, R2
    BLT  lrs_neon_scalar
    VLD1.P 16(R1), [V0.S4]
    VFCMGT_S4(8, 0, 4)
    VFMUL_S4(29, 0, 5)
    VBSL_B16(4, 0, 7)
    VST1.P [V7.S4], 16(R0)
    SUB  $4, R2
    B    lrs_neon_w4

lrs_neon_scalar:
    CBZ  R2, lrs_neon_done

lrs_neon_sloop:
    FMOVS (R1), F0
    VDUP V0.S[0], V0.S4
    VFCMGT_S4(8, 0, 4)
    VFMUL_S4(29, 0, 5)
    VBSL_B16(4, 0, 7)
    FMOVS F7, (R0)
    ADD  $4, R1
    ADD  $4, R0
    SUB  $1, R2
    CBNZ R2, lrs_neon_sloop

lrs_neon_done:
    RET

// func PReLUF32NEON(dst, src *float32, count int, negativeSlope float32)
TEXT ·PReLUF32NEON(SB), NOSPLIT, $0-28
    JMP ·LeakyReLUSlopeF32NEON(SB)

// func ThresholdF32NEON(dst, src *float32, count int, threshold float32)
TEXT ·ThresholdF32NEON(SB), NOSPLIT, $0-28
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD count+16(FP), R2
    FMOVS threshold+24(FP), F29
    VDUP V29.S[0], V29.S4

thr_neon_w4:
    CMP  $4, R2
    BLT  thr_neon_scalar
    VLD1.P 16(R1), [V0.S4]
    VFCMGT_S4(29, 0, 4)
    VBSL_B16(4, 0, 7)
    VST1.P [V7.S4], 16(R0)
    SUB  $4, R2
    B    thr_neon_w4

thr_neon_scalar:
    CBZ  R2, thr_neon_done

thr_neon_sloop:
    FMOVS (R1), F0
    VDUP V0.S[0], V0.S4
    VFCMGT_S4(29, 0, 4)
    VBSL_B16(4, 0, 7)
    FMOVS F7, (R0)
    ADD  $4, R1
    ADD  $4, R0
    SUB  $1, R2
    CBNZ R2, thr_neon_sloop

thr_neon_done:
    RET

// func HardTanhRangeF32NEON(dst, src *float32, count int, minVal, maxVal float32)
TEXT ·HardTanhRangeF32NEON(SB), NOSPLIT, $0-32
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD count+16(FP), R2
    FMOVS minVal+24(FP), F28
    VDUP V28.S[0], V28.S4
    FMOVS maxVal+28(FP), F29
    VDUP V29.S[0], V29.S4

htr_neon_w4:
    CMP  $4, R2
    BLT  htr_neon_scalar
    VLD1.P 16(R1), [V0.S4]
    VMAX_S4(28, 0, 7)
    VMIN_S4(29, 7, 7)
    VST1.P [V7.S4], 16(R0)
    SUB  $4, R2
    B    htr_neon_w4

htr_neon_scalar:
    CBZ  R2, htr_neon_done

htr_neon_sloop:
    FMOVS (R1), F0
    VDUP V0.S[0], V0.S4
    VMAX_S4(28, 0, 7)
    VMIN_S4(29, 7, 7)
    FMOVS F7, (R0)
    ADD  $4, R1
    ADD  $4, R0
    SUB  $1, R2
    CBNZ R2, htr_neon_sloop

htr_neon_done:
    RET
