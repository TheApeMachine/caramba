// SPDX-License-Identifier: Apache-2.0
// NEON kernels for remaining activations.


#include "textflag.h"
#define VFADD_S4(m, n, d)  WORD $(0x4E20D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_S4(m, n, d)  WORD $(0x4EA0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_S4(m, n, d)  WORD $(0x6E20DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFMLA_S4(m, n, d)  WORD $(0x4E20CC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFDIV_S4(m, n, d)  WORD $(0x6E20FC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFRINTN_S4(n, d)   WORD $(0x4E218800 | ((n) << 5) | (d))
#define VFCVTZS_S4(n, d)   WORD $(0x4EA1B800 | ((n) << 5) | (d))
#define VADD_S4(m, n, d)   WORD $(0x4EA08400 | ((m) << 16) | ((n) << 5) | (d))
#define VSHL_S4_BY23(n, d) WORD $(0x4F375400 | ((n) << 5) | (d))
#define VMOV_B16(src, dst) WORD $(0x4EA01C00 | ((src) << 16) | ((src) << 5) | (dst))
#define VFCMGT_S4(m, n, d) WORD $(0x6EA0E400 | ((m) << 16) | ((n) << 5) | (d))
#define VFCMGE_S4(m, n, d) WORD $(0x6EA0C400 | ((m) << 16) | ((n) << 5) | (d))
#define VFCMLE_S4(m, n, d) WORD $(0x6EA0E000 | ((m) << 16) | ((n) << 5) | (d))
#define VBSL_B16(m, n, d)  WORD $(0x6E601C00 | ((m) << 16) | ((n) << 5) | (d))
#define VFNEG_S4(n, d)     WORD $(0x6EA0F800 | ((n) << 5) | (d))
#define VUSHR_S4_BY23(n, d) WORD $(0x6F290400 | ((n) << 5) | (d))
#define VISUB_S4(m, n, d)   WORD $(0x6EA08400 | ((m) << 16) | ((n) << 5) | (d))
#define VAND_B16(m, n, d)   WORD $(0x4E201C00 | ((m) << 16) | ((n) << 5) | (d))
#define VORR_B16(m, n, d)   WORD $(0x4EA01C00 | ((m) << 16) | ((n) << 5) | (d))
#define VSCVTF_S4(n, d)     WORD $(0x4E21D800 | ((n) << 5) | (d))
#define VFABS_S4(n, d)      WORD $(0x6EA0F000 | ((n) << 5) | (d))
#define VMAX_S4(m, n, d)    WORD $(0x6EA0F400 | ((m) << 16) | ((n) << 5) | (d))
#define VMIN_S4(m, n, d)    WORD $(0x6EA0D400 | ((m) << 16) | ((n) << 5) | (d))


DATA actExtraExpC+0(SB)/4, $1.4426950408889634
DATA actExtraExpC+4(SB)/4, $0.6931471805599453
DATA actExtraExpC+8(SB)/4, $127.0
DATA actExtraExpC+12(SB)/4, $0.00019841270
DATA actExtraExpC+16(SB)/4, $0.0013888889
DATA actExtraExpC+20(SB)/4, $0.008333334
DATA actExtraExpC+24(SB)/4, $0.041666667
DATA actExtraExpC+28(SB)/4, $0.16666667
DATA actExtraExpC+32(SB)/4, $0.5
DATA actExtraExpC+36(SB)/4, $1.0
DATA actExtraExpC+40(SB)/4, $1.0
DATA actExtraExpC+44(SB)/4, $2.0
GLOBL actExtraExpC(SB), 8, $48


DATA actExtraLogC+0(SB)/4, $0.6931471805599453
DATA actExtraLogC+4(SB)/4, $1.0
DATA actExtraLogC+8(SB)/4, $0.09090909
DATA actExtraLogC+12(SB)/4, $0.11111111
DATA actExtraLogC+16(SB)/4, $0.14285715
DATA actExtraLogC+20(SB)/4, $0.20000000
DATA actExtraLogC+24(SB)/4, $0.33333334
DATA actExtraLogC+28(SB)/4, $2.0
GLOBL actExtraLogC(SB), 8, $32


DATA actExtraMiscC+0(SB)/4, $0.01
DATA actExtraMiscC+4(SB)/4, $1.6732632423543772
DATA actExtraMiscC+8(SB)/4, $1.0507009873554805
DATA actExtraMiscC+12(SB)/4, $20.0
DATA actExtraMiscC+16(SB)/4, $0.5
DATA actExtraMiscC+20(SB)/4, $6.0
DATA actExtraMiscC+24(SB)/4, $3.0
DATA actExtraMiscC+28(SB)/4, $-1.0
DATA actExtraMiscC+32(SB)/4, $1.702
DATA actExtraMiscC+36(SB)/4, $0.7071067811865475
DATA actExtraMiscC+40(SB)/4, $135135.0
DATA actExtraMiscC+44(SB)/4, $17325.0
DATA actExtraMiscC+48(SB)/4, $378.0
DATA actExtraMiscC+52(SB)/4, $62370.0
DATA actExtraMiscC+56(SB)/4, $3150.0
DATA actExtraMiscC+60(SB)/4, $28.0
DATA actExtraMiscC+64(SB)/4, $4.92
DATA actExtraMiscC+68(SB)/4, $-4.92
DATA actExtraMiscC+72(SB)/4, $0.7978845608028654
DATA actExtraMiscC+76(SB)/4, $0.044715
GLOBL actExtraMiscC(SB), 8, $80


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


#define NEON_LOG_BODY(in, out) \
    VUSHR_S4_BY23(in, 1) ;\
    VISUB_S4(26, 1, 1) ;\
    VAND_B16(24, in, 2) ;\
    VORR_B16(25, 2, 2) ;\
    VSCVTF_S4(1, 1) ;\
    VFSUB_S4(17, 2, 3) ;\
    VFADD_S4(17, 2, 4) ;\
    VFDIV_S4(4, 3, 5) ;\
    VFMUL_S4(5, 5, 6) ;\
    VMOV_B16(18, 7) ;\
    VMOV_B16(19, 8) ; VFMLA_S4(6, 7, 8) ;\
    VMOV_B16(20, 7) ; VFMLA_S4(6, 8, 7) ;\
    VMOV_B16(21, 8) ; VFMLA_S4(6, 7, 8) ;\
    VMOV_B16(22, 7) ; VFMLA_S4(6, 8, 7) ;\
    VMOV_B16(17, 8) ; VFMLA_S4(6, 7, 8) ;\
    VFMUL_S4(5, 8, 8) ;\
    VFMUL_S4(23, 8, 8) ;\
    VFMLA_S4(16, 1, out)


#define NEON_TANH_PADÉ(xreg, outreg) \
    VFMUL_S4(xreg, xreg, 1) ;\
    FMOVS 40(R3), F29 ; VDUP V29.S[0], V29.S4 ;\
    FMOVS 44(R3), F30 ; VDUP V30.S[0], V30.S4 ;\
    FMOVS 48(R3), F31 ; VDUP V31.S[0], V31.S4 ;\
    VFMLA_S4(1, 30, 2) ;\
    VFMLA_S4(1, 31, 2) ;\
    VFMUL_S4(xreg, 2, 2) ;\
    FMOVS 52(R3), F30 ; VDUP V30.S[0], V30.S4 ;\
    FMOVS 56(R3), F31 ; VDUP V31.S[0], V31.S4 ;\
    VFMLA_S4(1, 30, 3) ;\
    VFMLA_S4(1, 31, 3) ;\
    VFDIV_S4(3, 2, outreg) ;\
    FMOVS 64(R3), F29 ; VDUP V29.S[0], V29.S4 ;\
    FMOVS 68(R3), F30 ; VDUP V30.S[0], V30.S4 ;\
    VMAX_S4(30, outreg, outreg) ;\
    VMIN_S4(29, outreg, outreg)


// func Log1pF32NEON(dst, src *float32, count int)
TEXT ·Log1pF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2

MOVD $actExtraLogC(SB), R3
FMOVS  0(R3), F16
FMOVS  4(R3), F17
FMOVS  8(R3), F18
FMOVS 12(R3), F19
FMOVS 16(R3), F20
FMOVS 20(R3), F21
FMOVS 24(R3), F22
FMOVS 28(R3), F23
MOVD $0x007FFFFF, R6
VMOV R6, V24.S[0]
VDUP V24.S[0], V24.S4
MOVD $0x3F800000, R6
VMOV R6, V25.S[0]
VDUP V25.S[0], V25.S4
MOVD $127, R6
VMOV R6, V26.S[0]
VDUP V26.S[0], V26.S4

    loop4:
        CMP  $4, R2
        BLT  scalar_tail
        VLD1.P 16(R1), [V0.S4]
        VFADD_S4(17, 0, 0)
NEON_LOG_BODY(0, 7)
        VST1.P [V7.S4], 16(R0)
        SUB  $4, R2
        B    loop4
    scalar_tail:
        CBZ  R2, done
    scalar_loop:
        FMOVS (R1), F0
        VDUP V0.S[0], V0.S4
        VFADD_S4(17, 0, 0)
NEON_LOG_BODY(0, 7)
        FMOVS F7, (R0)
        ADD  $4, R1
        ADD  $4, R0
        SUB  $1, R2
        CBNZ R2, scalar_loop
    done:
        RET


// func Expm1F32NEON(dst, src *float32, count int)
TEXT ·Expm1F32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2

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

    loop4:
        CMP  $4, R2
        BLT  scalar_tail
        VLD1.P 16(R1), [V0.S4]
        NEON_EXP_BODY(0, 6)
VFSUB_S4(26, 6, 7)
        VST1.P [V7.S4], 16(R0)
        SUB  $4, R2
        B    loop4
    scalar_tail:
        CBZ  R2, done
    scalar_loop:
        FMOVS (R1), F0
        VDUP V0.S[0], V0.S4
        NEON_EXP_BODY(0, 6)
VFSUB_S4(26, 6, 7)
        FMOVS F7, (R0)
        ADD  $4, R1
        ADD  $4, R0
        SUB  $1, R2
        CBNZ R2, scalar_loop
    done:
        RET


// func LogSigmoidF32NEON(dst, src *float32, count int)
TEXT ·LogSigmoidF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2

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

MOVD $actExtraLogC(SB), R3
FMOVS  0(R3), F16
FMOVS  4(R3), F17
FMOVS  8(R3), F18
FMOVS 12(R3), F19
FMOVS 16(R3), F20
FMOVS 20(R3), F21
FMOVS 24(R3), F22
FMOVS 28(R3), F23
MOVD $0x007FFFFF, R6
VMOV R6, V24.S[0]
VDUP V24.S[0], V24.S4
MOVD $0x3F800000, R6
VMOV R6, V25.S[0]
VDUP V25.S[0], V25.S4
MOVD $127, R6
VMOV R6, V26.S[0]
VDUP V26.S[0], V26.S4

        loop4:
            CMP  $4, R2
            BLT  scalar_tail
            VLD1.P 16(R1), [V0.S4]

VFNEG_S4(0, 0)
VMOV_B16(0, 8)
VFNEG_S4(0, 0)
NEON_EXP_BODY(0, 6)
VFADD_S4(26, 6, 6)
VFADD_S4(17, 6, 6)
NEON_LOG_BODY(6, 7)
VFNEG_S4(7, 7)

            VST1.P [V7.S4], 16(R0)
            SUB  $4, R2
            B    loop4
        scalar_tail:
            CBZ  R2, done
        scalar_loop:
            FMOVS (R1), F0
            VDUP V0.S[0], V0.S4

VFNEG_S4(0, 0)
VMOV_B16(0, 8)
VFNEG_S4(0, 0)
NEON_EXP_BODY(0, 6)
VFADD_S4(26, 6, 6)
VFADD_S4(17, 6, 6)
NEON_LOG_BODY(6, 7)
VFNEG_S4(7, 7)

            FMOVS F7, (R0)
            ADD  $4, R1
            ADD  $4, R0
            SUB  $1, R2
            CBNZ R2, scalar_loop
        done:
            RET


// func GeluF32NEON(dst, src *float32, count int)
TEXT ·GeluF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2

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
    MOVD $actExtraMiscC(SB), R3
    FMOVS 32(R3), F24

        loop4:
            CMP  $4, R2
            BLT  scalar_tail
            VLD1.P 16(R1), [V0.S4]

VMOV_B16(0, 10)
FMOVS 36(R3), F29
VFMUL_S4(29, 0, 11)
VFABS_S4(11, 12)
FMOVS 0(R3), F30
VFADD_S4(30, 12, 13)
VFDIV_S4(13, 12, 12)
VFNEG_S4(11, 14)
VFMUL_S4(14, 14, 15)
VFMLA_S4(11, 12, 14)
VFMUL_S4(14, 12, 14)
VFSUB_S4(26, 14, 14)
VFCMGT_S4(8, 11, 13)
VFNEG_S4(14, 15)
VBSL_B16(13, 15, 14)
VFADD_S4(26, 14, 14)
VFMUL_S4(10, 14, 7)
VFMUL_S4(24, 7, 7)

            VST1.P [V7.S4], 16(R0)
            SUB  $4, R2
            B    loop4
        scalar_tail:
            CBZ  R2, done
        scalar_loop:
            FMOVS (R1), F0
            VDUP V0.S[0], V0.S4

VMOV_B16(0, 10)
FMOVS 36(R3), F29
VFMUL_S4(29, 0, 11)
VFABS_S4(11, 12)
FMOVS 0(R3), F30
VFADD_S4(30, 12, 13)
VFDIV_S4(13, 12, 12)
VFNEG_S4(11, 14)
VFMUL_S4(14, 14, 15)
VFMLA_S4(11, 12, 14)
VFMUL_S4(14, 12, 14)
VFSUB_S4(26, 14, 14)
VFCMGT_S4(8, 11, 13)
VFNEG_S4(14, 15)
VBSL_B16(13, 15, 14)
VFADD_S4(26, 14, 14)
VFMUL_S4(10, 14, 7)
VFMUL_S4(24, 7, 7)

            FMOVS F7, (R0)
            ADD  $4, R1
            ADD  $4, R0
            SUB  $1, R2
            CBNZ R2, scalar_loop
        done:
            RET


// func LeakyReLUF32NEON(dst, src *float32, count int)
TEXT ·LeakyReLUF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2
    MOVD $actExtraMiscC(SB), R3

VEOR V8.B16, V8.B16, V8.B16
FMOVS 0(R3), F29
VDUP V29.S[0], V29.S4

    loop4:
        CMP  $4, R2
        BLT  scalar_tail
        VLD1.P 16(R1), [V0.S4]
        VFCMGT_S4(8, 0, 4)
VFMUL_S4(29, 0, 5)
VBSL_B16(4, 0, 7)
        VST1.P [V7.S4], 16(R0)
        SUB  $4, R2
        B    loop4
    scalar_tail:
        CBZ  R2, done
    scalar_loop:
        FMOVS (R1), F0
        VDUP V0.S[0], V0.S4
        VFCMGT_S4(8, 0, 4)
VFMUL_S4(29, 0, 5)
VBSL_B16(4, 0, 7)
        FMOVS F7, (R0)
        ADD  $4, R1
        ADD  $4, R0
        SUB  $1, R2
        CBNZ R2, scalar_loop
    done:
        RET


// func ELUF32NEON(dst, src *float32, count int)
TEXT ·ELUF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2

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

    loop4:
        CMP  $4, R2
        BLT  scalar_tail
        VLD1.P 16(R1), [V0.S4]
        VFCMGT_S4(8, 0, 4)
VMOV_B16(0, 10)
NEON_EXP_BODY(0, 6)
VFSUB_S4(26, 6, 6)
VBSL_B16(4, 10, 7)
        VST1.P [V7.S4], 16(R0)
        SUB  $4, R2
        B    loop4
    scalar_tail:
        CBZ  R2, done
    scalar_loop:
        FMOVS (R1), F0
        VDUP V0.S[0], V0.S4
        VFCMGT_S4(8, 0, 4)
VMOV_B16(0, 10)
NEON_EXP_BODY(0, 6)
VFSUB_S4(26, 6, 6)
VBSL_B16(4, 10, 7)
        FMOVS F7, (R0)
        ADD  $4, R1
        ADD  $4, R0
        SUB  $1, R2
        CBNZ R2, scalar_loop
    done:
        RET


// func CELUF32NEON(dst, src *float32, count int)
TEXT ·CELUF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2

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

    loop4:
        CMP  $4, R2
        BLT  scalar_tail
        VLD1.P 16(R1), [V0.S4]
        VFCMGT_S4(8, 0, 4)
VMOV_B16(0, 10)
NEON_EXP_BODY(0, 6)
VFSUB_S4(26, 6, 6)
VBSL_B16(4, 10, 7)
        VST1.P [V7.S4], 16(R0)
        SUB  $4, R2
        B    loop4
    scalar_tail:
        CBZ  R2, done
    scalar_loop:
        FMOVS (R1), F0
        VDUP V0.S[0], V0.S4
        VFCMGT_S4(8, 0, 4)
VMOV_B16(0, 10)
NEON_EXP_BODY(0, 6)
VFSUB_S4(26, 6, 6)
VBSL_B16(4, 10, 7)
        FMOVS F7, (R0)
        ADD  $4, R1
        ADD  $4, R0
        SUB  $1, R2
        CBNZ R2, scalar_loop
    done:
        RET


// func SELUF32NEON(dst, src *float32, count int)
TEXT ·SELUF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2

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
    MOVD $actExtraMiscC(SB), R3

VEOR V8.B16, V8.B16, V8.B16

    loop4:
        CMP  $4, R2
        BLT  scalar_tail
        VLD1.P 16(R1), [V0.S4]
        VFCMGT_S4(8, 0, 4)
FMOVS 8(R3), F29
VFMUL_S4(29, 0, 5)
NEON_EXP_BODY(0, 6)
VFSUB_S4(26, 6, 6)
FMOVS 4(R3), F30
VFMUL_S4(30, 6, 6)
VBSL_B16(4, 5, 7)
        VST1.P [V7.S4], 16(R0)
        SUB  $4, R2
        B    loop4
    scalar_tail:
        CBZ  R2, done
    scalar_loop:
        FMOVS (R1), F0
        VDUP V0.S[0], V0.S4
        VFCMGT_S4(8, 0, 4)
FMOVS 8(R3), F29
VFMUL_S4(29, 0, 5)
NEON_EXP_BODY(0, 6)
VFSUB_S4(26, 6, 6)
FMOVS 4(R3), F30
VFMUL_S4(30, 6, 6)
VBSL_B16(4, 5, 7)
        FMOVS F7, (R0)
        ADD  $4, R1
        ADD  $4, R0
        SUB  $1, R2
        CBNZ R2, scalar_loop
    done:
        RET


// func SoftplusF32NEON(dst, src *float32, count int)
TEXT ·SoftplusF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2

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

MOVD $actExtraLogC(SB), R3
FMOVS  0(R3), F16
FMOVS  4(R3), F17
FMOVS  8(R3), F18
FMOVS 12(R3), F19
FMOVS 16(R3), F20
FMOVS 20(R3), F21
FMOVS 24(R3), F22
FMOVS 28(R3), F23
MOVD $0x007FFFFF, R6
VMOV R6, V24.S[0]
VDUP V24.S[0], V24.S4
MOVD $0x3F800000, R6
VMOV R6, V25.S[0]
VDUP V25.S[0], V25.S4
MOVD $127, R6
VMOV R6, V26.S[0]
VDUP V26.S[0], V26.S4
    MOVD $actExtraMiscC(SB), R3

    loop4:
        CMP  $4, R2
        BLT  scalar_tail
        VLD1.P 16(R1), [V0.S4]
        FMOVS 12(R3), F29
VFCMGT_S4(29, 0, 4)
VMOV_B16(0, 7)
VFNEG_S4(0, 0)
NEON_EXP_BODY(0, 6)
VFADD_S4(26, 6, 6)
NEON_LOG_BODY(6, 5)
VBSL_B16(4, 7, 5)
        VST1.P [V7.S4], 16(R0)
        SUB  $4, R2
        B    loop4
    scalar_tail:
        CBZ  R2, done
    scalar_loop:
        FMOVS (R1), F0
        VDUP V0.S[0], V0.S4
        FMOVS 12(R3), F29
VFCMGT_S4(29, 0, 4)
VMOV_B16(0, 7)
VFNEG_S4(0, 0)
NEON_EXP_BODY(0, 6)
VFADD_S4(26, 6, 6)
NEON_LOG_BODY(6, 5)
VBSL_B16(4, 7, 5)
        FMOVS F7, (R0)
        ADD  $4, R1
        ADD  $4, R0
        SUB  $1, R2
        CBNZ R2, scalar_loop
    done:
        RET


// func MishF32NEON(dst, src *float32, count int)
TEXT ·MishF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2

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

MOVD $actExtraLogC(SB), R3
FMOVS  0(R3), F16
FMOVS  4(R3), F17
FMOVS  8(R3), F18
FMOVS 12(R3), F19
FMOVS 16(R3), F20
FMOVS 20(R3), F21
FMOVS 24(R3), F22
FMOVS 28(R3), F23
MOVD $0x007FFFFF, R6
VMOV R6, V24.S[0]
VDUP V24.S[0], V24.S4
MOVD $0x3F800000, R6
VMOV R6, V25.S[0]
VDUP V25.S[0], V25.S4
MOVD $127, R6
VMOV R6, V26.S[0]
VDUP V26.S[0], V26.S4
    MOVD $actExtraMiscC(SB), R3

    loop4:
        CMP  $4, R2
        BLT  scalar_tail
        VLD1.P 16(R1), [V0.S4]
        VMOV_B16(0, 10)
FMOVS 12(R3), F29
VFCMGT_S4(29, 0, 4)
VMOV_B16(0, 7)
VFNEG_S4(0, 0)
NEON_EXP_BODY(0, 6)
VFADD_S4(26, 6, 6)
NEON_LOG_BODY(6, 5)
VBSL_B16(4, 7, 5)
NEON_TANH_PADÉ(5, 6)
VFMUL_S4(10, 6, 7)
        VST1.P [V7.S4], 16(R0)
        SUB  $4, R2
        B    loop4
    scalar_tail:
        CBZ  R2, done
    scalar_loop:
        FMOVS (R1), F0
        VDUP V0.S[0], V0.S4
        VMOV_B16(0, 10)
FMOVS 12(R3), F29
VFCMGT_S4(29, 0, 4)
VMOV_B16(0, 7)
VFNEG_S4(0, 0)
NEON_EXP_BODY(0, 6)
VFADD_S4(26, 6, 6)
NEON_LOG_BODY(6, 5)
VBSL_B16(4, 7, 5)
NEON_TANH_PADÉ(5, 6)
VFMUL_S4(10, 6, 7)
        FMOVS F7, (R0)
        ADD  $4, R1
        ADD  $4, R0
        SUB  $1, R2
        CBNZ R2, scalar_loop
    done:
        RET


// func SoftsignF32NEON(dst, src *float32, count int)
TEXT ·SoftsignF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2
    VEOR V8.B16, V8.B16, V8.B16
    MOVD $actExtraExpC(SB), R3
    FMOVS 36(R3), F26
    VDUP V26.S[0], V26.S4

        loop4:
            CMP  $4, R2
            BLT  scalar_tail
            VLD1.P 16(R1), [V0.S4]

VFABS_S4(0, 1)
VFADD_S4(26, 1, 1)
VFDIV_S4(1, 0, 7)
VFCMGT_S4(8, 0, 4)
VFNEG_S4(7, 5)
VBSL_B16(4, 5, 7)

            VST1.P [V7.S4], 16(R0)
            SUB  $4, R2
            B    loop4
        scalar_tail:
            CBZ  R2, done
        scalar_loop:
            FMOVS (R1), F0
            VDUP V0.S[0], V0.S4

VFABS_S4(0, 1)
VFADD_S4(26, 1, 1)
VFDIV_S4(1, 0, 7)
VFCMGT_S4(8, 0, 4)
VFNEG_S4(7, 5)
VBSL_B16(4, 5, 7)

            FMOVS F7, (R0)
            ADD  $4, R1
            ADD  $4, R0
            SUB  $1, R2
            CBNZ R2, scalar_loop
        done:
            RET


// func HardSigmoidF32NEON(dst, src *float32, count int)
TEXT ·HardSigmoidF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2
    MOVD $actExtraMiscC(SB), R3

VEOR V8.B16, V8.B16, V8.B16

    loop4:
        CMP  $4, R2
        BLT  scalar_tail
        VLD1.P 16(R1), [V0.S4]
        FMOVS 16(R3), F29
VFMUL_S4(29, 0, 7)
VFADD_S4(29, 7, 7)
VEOR V8.B16, V8.B16, V8.B16
VFCMGT_S4(8, 7, 4)
VBSL_B16(4, 7, 7)
FMOVS 4(R3), F30
VFCMLE_S4(30, 7, 4)
FMOVS 26(R3), F30
VBSL_B16(4, 30, 7)
        VST1.P [V7.S4], 16(R0)
        SUB  $4, R2
        B    loop4
    scalar_tail:
        CBZ  R2, done
    scalar_loop:
        FMOVS (R1), F0
        VDUP V0.S[0], V0.S4
        FMOVS 16(R3), F29
VFMUL_S4(29, 0, 7)
VFADD_S4(29, 7, 7)
VEOR V8.B16, V8.B16, V8.B16
VFCMGT_S4(8, 7, 4)
VBSL_B16(4, 7, 7)
FMOVS 4(R3), F30
VFCMLE_S4(30, 7, 4)
FMOVS 26(R3), F30
VBSL_B16(4, 30, 7)
        FMOVS F7, (R0)
        ADD  $4, R1
        ADD  $4, R0
        SUB  $1, R2
        CBNZ R2, scalar_loop
    done:
        RET


// func HardSwishF32NEON(dst, src *float32, count int)
TEXT ·HardSwishF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2
    MOVD $actExtraMiscC(SB), R3

VEOR V8.B16, V8.B16, V8.B16

    loop4:
        CMP  $4, R2
        BLT  scalar_tail
        VLD1.P 16(R1), [V0.S4]
        VMOV_B16(0, 10)
FMOVS 24(R3), F29
VFADD_S4(29, 0, 0)
FMOVS 16(R3), F29
VFMUL_S4(29, 0, 7)
VFADD_S4(29, 7, 7)
VEOR V8.B16, V8.B16, V8.B16
VFCMGT_S4(8, 7, 4)
VBSL_B16(4, 7, 7)
FMOVS 4(R3), F30
VFCMLE_S4(30, 7, 4)
FMOVS 26(R3), F30
VBSL_B16(4, 30, 7)
VFMUL_S4(10, 7, 7)
        VST1.P [V7.S4], 16(R0)
        SUB  $4, R2
        B    loop4
    scalar_tail:
        CBZ  R2, done
    scalar_loop:
        FMOVS (R1), F0
        VDUP V0.S[0], V0.S4
        VMOV_B16(0, 10)
FMOVS 24(R3), F29
VFADD_S4(29, 0, 0)
FMOVS 16(R3), F29
VFMUL_S4(29, 0, 7)
VFADD_S4(29, 7, 7)
VEOR V8.B16, V8.B16, V8.B16
VFCMGT_S4(8, 7, 4)
VBSL_B16(4, 7, 7)
FMOVS 4(R3), F30
VFCMLE_S4(30, 7, 4)
FMOVS 26(R3), F30
VBSL_B16(4, 30, 7)
VFMUL_S4(10, 7, 7)
        FMOVS F7, (R0)
        ADD  $4, R1
        ADD  $4, R0
        SUB  $1, R2
        CBNZ R2, scalar_loop
    done:
        RET


// func HardTanhF32NEON(dst, src *float32, count int)
TEXT ·HardTanhF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2
    MOVD $actExtraMiscC(SB), R3

    loop4:
        CMP  $4, R2
        BLT  scalar_tail
        VLD1.P 16(R1), [V0.S4]
        FMOVS 28(R3), F29
VMAX_S4(29, 0, 7)
FMOVS 4(R3), F30
VMIN_S4(30, 7, 7)
        VST1.P [V7.S4], 16(R0)
        SUB  $4, R2
        B    loop4
    scalar_tail:
        CBZ  R2, done
    scalar_loop:
        FMOVS (R1), F0
        VDUP V0.S[0], V0.S4
        FMOVS 28(R3), F29
VMAX_S4(29, 0, 7)
FMOVS 4(R3), F30
VMIN_S4(30, 7, 7)
        FMOVS F7, (R0)
        ADD  $4, R1
        ADD  $4, R0
        SUB  $1, R2
        CBNZ R2, scalar_loop
    done:
        RET


// func HardGeluF32NEON(dst, src *float32, count int)
TEXT ·HardGeluF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2
    MOVD $actExtraMiscC(SB), R3

VMOV_B16(0, 10)
FMOVS 24(R3), F29
VFADD_S4(29, 0, 7)
VEOR V8.B16, V8.B16, V8.B16
VFCMGT_S4(8, 7, 4)
VBSL_B16(4, 7, 7)
FMOVS 20(R3), F30
VFCMLE_S4(30, 7, 4)
VBSL_B16(4, 30, 7)
FMOVS 16(R3), F29
VFDIV_S4(29, 7, 7)
VFMUL_S4(10, 7, 7)

    loop4:
        CMP  $4, R2
        BLT  scalar_tail
        VLD1.P 16(R1), [V0.S4]
        VMOV_B16(0, 10)
FMOVS 24(R3), F29
VFADD_S4(29, 0, 7)
VEOR V8.B16, V8.B16, V8.B16
VFCMGT_S4(8, 7, 4)
VBSL_B16(4, 7, 7)
FMOVS 20(R3), F30
VFCMLE_S4(30, 7, 4)
VBSL_B16(4, 30, 7)
FMOVS 16(R3), F29
VFDIV_S4(29, 7, 7)
VFMUL_S4(10, 7, 7)
        VST1.P [V7.S4], 16(R0)
        SUB  $4, R2
        B    loop4
    scalar_tail:
        CBZ  R2, done
    scalar_loop:
        FMOVS (R1), F0
        VDUP V0.S[0], V0.S4
        VMOV_B16(0, 10)
FMOVS 24(R3), F29
VFADD_S4(29, 0, 7)
VEOR V8.B16, V8.B16, V8.B16
VFCMGT_S4(8, 7, 4)
VBSL_B16(4, 7, 7)
FMOVS 20(R3), F30
VFCMLE_S4(30, 7, 4)
VBSL_B16(4, 30, 7)
FMOVS 16(R3), F29
VFDIV_S4(29, 7, 7)
VFMUL_S4(10, 7, 7)
        FMOVS F7, (R0)
        ADD  $4, R1
        ADD  $4, R0
        SUB  $1, R2
        CBNZ R2, scalar_loop
    done:
        RET


// func QuickGeluF32NEON(dst, src *float32, count int)
TEXT ·QuickGeluF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2

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
    MOVD $actExtraMiscC(SB), R3

VMOV_B16(0, 10)
FMOVS 32(R3), F29
VFMUL_S4(29, 0, 0)
VFNEG_S4(0, 0)
NEON_EXP_BODY(0, 6)
VFADD_S4(26, 6, 6)
VFDIV_S4(6, 26, 7)
VFMUL_S4(10, 7, 7)

    loop4:
        CMP  $4, R2
        BLT  scalar_tail
        VLD1.P 16(R1), [V0.S4]
        VMOV_B16(0, 10)
FMOVS 32(R3), F29
VFMUL_S4(29, 0, 0)
VFNEG_S4(0, 0)
NEON_EXP_BODY(0, 6)
VFADD_S4(26, 6, 6)
VFDIV_S4(6, 26, 7)
VFMUL_S4(10, 7, 7)
        VST1.P [V7.S4], 16(R0)
        SUB  $4, R2
        B    loop4
    scalar_tail:
        CBZ  R2, done
    scalar_loop:
        FMOVS (R1), F0
        VDUP V0.S[0], V0.S4
        VMOV_B16(0, 10)
FMOVS 32(R3), F29
VFMUL_S4(29, 0, 0)
VFNEG_S4(0, 0)
NEON_EXP_BODY(0, 6)
VFADD_S4(26, 6, 6)
VFDIV_S4(6, 26, 7)
VFMUL_S4(10, 7, 7)
        FMOVS F7, (R0)
        ADD  $4, R1
        ADD  $4, R0
        SUB  $1, R2
        CBNZ R2, scalar_loop
    done:
        RET


// func TanhShrinkF32NEON(dst, src *float32, count int)
TEXT ·TanhShrinkF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2
    MOVD $actExtraMiscC(SB), R3

VMOV_B16(0, 10)
NEON_TANH_PADÉ(0, 6)
VFSUB_S4(6, 10, 7)

    loop4:
        CMP  $4, R2
        BLT  scalar_tail
        VLD1.P 16(R1), [V0.S4]
        VMOV_B16(0, 10)
NEON_TANH_PADÉ(0, 6)
VFSUB_S4(6, 10, 7)
        VST1.P [V7.S4], 16(R0)
        SUB  $4, R2
        B    loop4
    scalar_tail:
        CBZ  R2, done
    scalar_loop:
        FMOVS (R1), F0
        VDUP V0.S[0], V0.S4
        VMOV_B16(0, 10)
NEON_TANH_PADÉ(0, 6)
VFSUB_S4(6, 10, 7)
        FMOVS F7, (R0)
        ADD  $4, R1
        ADD  $4, R0
        SUB  $1, R2
        CBNZ R2, scalar_loop
    done:
        RET
