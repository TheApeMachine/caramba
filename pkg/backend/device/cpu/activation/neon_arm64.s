// SPDX-License-Identifier: Apache-2.0
// NEON float32 activation kernels for package activation.
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

// Constants — laid out so a single broadcast can pick each one up.
DATA  actExpC<>+0(SB)/4, $1.4426950408889634   // log2e
DATA  actExpC<>+4(SB)/4, $0.6931471805599453   // ln2
DATA  actExpC<>+8(SB)/4, $127.0                // bias as float (for VFCVTZS to 127)
DATA  actExpC<>+12(SB)/4, $0.00019841270       // c7 = 1/5040
DATA  actExpC<>+16(SB)/4, $0.0013888889        // c6 = 1/720
DATA  actExpC<>+20(SB)/4, $0.008333334         // c5 = 1/120
DATA  actExpC<>+24(SB)/4, $0.041666667         // c4 = 1/24
DATA  actExpC<>+28(SB)/4, $0.16666667          // c3 = 1/6
DATA  actExpC<>+32(SB)/4, $0.5                 // c2
DATA  actExpC<>+36(SB)/4, $1.0                 // c1
DATA  actExpC<>+40(SB)/4, $1.0                 // c0
GLOBL actExpC<>(SB), 8, $44

// func ExpFloat32NEONAsm(dst, src *float32, n int)
TEXT ·ExpF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2

    MOVD $actExpC<>(SB), R3
    // Broadcast constants into V16..V26.
    FMOVS  0(R3), F16  ; VDUP V16.S[0], V16.S4  // log2e
    FMOVS  4(R3), F17  ; VDUP V17.S[0], V17.S4  // ln2
    FMOVS  8(R3), F18  ; VDUP V18.S[0], V18.S4  // 127.0
    FMOVS 12(R3), F19  ; VDUP V19.S[0], V19.S4  // c7
    FMOVS 16(R3), F20  ; VDUP V20.S[0], V20.S4  // c6
    FMOVS 20(R3), F21  ; VDUP V21.S[0], V21.S4  // c5
    FMOVS 24(R3), F22  ; VDUP V22.S[0], V22.S4  // c4
    FMOVS 28(R3), F23  ; VDUP V23.S[0], V23.S4  // c3
    FMOVS 32(R3), F24  ; VDUP V24.S[0], V24.S4  // c2
    FMOVS 36(R3), F25  ; VDUP V25.S[0], V25.S4  // c1
    FMOVS 40(R3), F26  ; VDUP V26.S[0], V26.S4  // c0
    VFCVTZS_S4(18, 27)                            // V27 = int32 127

exp_loop4:
    CMP  $4, R2
    BLT  exp_scalar_tail
    VLD1.P 16(R1), [V0.S4]
    // n_f = round(x * log2e)
    VFMUL_S4(16, 0, 1)                            // V1 = x * log2e
    VFRINTN_S4(1, 1)                              // V1 = round-to-nearest-even
    // r = x - n_f * ln2
    VFMUL_S4(17, 1, 2)                            // V2 = n_f * ln2
    VFSUB_S4(2, 0, 0)                             // V0 = x - V2 = r
    // Horner: y = c7
    //         y = c6 + r*y
    //         y = c5 + r*y
    //         ... (down to c0)
    VMOV_B16(19, 3)                               // V3 = c7
    VMOV_B16(20, 4) ; VFMLA_S4(0, 3, 4)           // V4 = c6 + r*V3
    VMOV_B16(21, 3) ; VFMLA_S4(0, 4, 3)           // V3 = c5 + r*V4
    VMOV_B16(22, 4) ; VFMLA_S4(0, 3, 4)           // V4 = c4 + r*V3
    VMOV_B16(23, 3) ; VFMLA_S4(0, 4, 3)           // V3 = c3 + r*V4
    VMOV_B16(24, 4) ; VFMLA_S4(0, 3, 4)           // V4 = c2 + r*V3
    VMOV_B16(25, 3) ; VFMLA_S4(0, 4, 3)           // V3 = c1 + r*V4
    VMOV_B16(26, 4) ; VFMLA_S4(0, 3, 4)           // V4 = c0 + r*V3 = exp(r)
    // 2^n = float32_from_bits((n_int+127) << 23)
    VFCVTZS_S4(1, 5)                              // V5 = n_int
    VADD_S4(27, 5, 5)                             // V5 += 127
    VSHL_S4_BY23(5, 5)                            // V5 = bits of 2^n
    VFMUL_S4(5, 4, 6)                             // V6 = exp(r) * 2^n
    VST1.P [V6.S4], 16(R0)
    SUB  $4, R2
    B    exp_loop4

exp_scalar_tail:
    CBZ  R2, exp_done

exp_scalar_loop:
    FMOVS (R1), F0
    VDUP V0.S[0], V0.S4
    VFMUL_S4(16, 0, 1)
    VFRINTN_S4(1, 1)
    VFMUL_S4(17, 1, 2)
    VFSUB_S4(2, 0, 0)
    VMOV_B16(19, 3)
    VMOV_B16(20, 4) ; VFMLA_S4(0, 3, 4)
    VMOV_B16(21, 3) ; VFMLA_S4(0, 4, 3)
    VMOV_B16(22, 4) ; VFMLA_S4(0, 3, 4)
    VMOV_B16(23, 3) ; VFMLA_S4(0, 4, 3)
    VMOV_B16(24, 4) ; VFMLA_S4(0, 3, 4)
    VMOV_B16(25, 3) ; VFMLA_S4(0, 4, 3)
    VMOV_B16(26, 4) ; VFMLA_S4(0, 3, 4)
    VFCVTZS_S4(1, 5)
    VADD_S4(27, 5, 5)
    VSHL_S4_BY23(5, 5)
    VFMUL_S4(5, 4, 6)
    FMOVS F6, (R0)
    ADD  $4, R1
    ADD  $4, R0
    SUB  $1, R2
    CBNZ R2, exp_scalar_loop

exp_done:
    RET

#define VUSHR_S4_BY23(n, d) WORD $(0x6F290400 | ((n) << 5) | (d))
#define VISUB_S4(m, n, d)   WORD $(0x6EA08400 | ((m) << 16) | ((n) << 5) | (d))
#define VAND_B16(m, n, d)   WORD $(0x4E201C00 | ((m) << 16) | ((n) << 5) | (d))
#define VORR_B16(m, n, d)   WORD $(0x4EA01C00 | ((m) << 16) | ((n) << 5) | (d))
#define VSCVTF_S4(n, d)     WORD $(0x4E21D800 | ((n) << 5) | (d))

DATA  actLogC<>+0(SB)/4, $0.6931471805599453   // ln2
DATA  actLogC<>+4(SB)/4, $1.0                  // 1.0
DATA  actLogC<>+8(SB)/4, $0.09090909           // 1/11
DATA  actLogC<>+12(SB)/4, $0.11111111          // 1/9
DATA  actLogC<>+16(SB)/4, $0.14285715          // 1/7
DATA  actLogC<>+20(SB)/4, $0.20000000          // 1/5
DATA  actLogC<>+24(SB)/4, $0.33333334          // 1/3
DATA  actLogC<>+28(SB)/4, $2.0                 // 2.0 (final scale)
GLOBL actLogC<>(SB), 8, $32

// Pre-built bitmasks. Loaded via integer reg then VMOV-broadcast.
//   exponentMask  = 0x7F800000   (used to extract exponent bits)
//   mantissaMask  = 0x007FFFFF
//   oneBits       = 0x3F800000   (bits of 1.0)
//   bias127       = 127 (int32)

// func LogFloat32NEONAsm(dst, src *float32, n int)
TEXT ·LogF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2

    MOVD $actLogC<>(SB), R3
    FMOVS  0(R3), F16  ; VDUP V16.S[0], V16.S4   // ln2
    FMOVS  4(R3), F17  ; VDUP V17.S[0], V17.S4   // 1.0
    FMOVS  8(R3), F18  ; VDUP V18.S[0], V18.S4   // 1/11
    FMOVS 12(R3), F19  ; VDUP V19.S[0], V19.S4   // 1/9
    FMOVS 16(R3), F20  ; VDUP V20.S[0], V20.S4   // 1/7
    FMOVS 20(R3), F21  ; VDUP V21.S[0], V21.S4   // 1/5
    FMOVS 24(R3), F22  ; VDUP V22.S[0], V22.S4   // 1/3
    FMOVS 28(R3), F23  ; VDUP V23.S[0], V23.S4   // 2.0

    // V24 = mantissa mask 0x007FFFFF broadcast
    MOVD $0x007FFFFF, R6
    VMOV R6, V24.S[0]
    VDUP V24.S[0], V24.S4
    // V25 = oneBits 0x3F800000 broadcast
    MOVD $0x3F800000, R6
    VMOV R6, V25.S[0]
    VDUP V25.S[0], V25.S4
    // V26 = bias 127 (int32) broadcast
    MOVD $127, R6
    VMOV R6, V26.S[0]
    VDUP V26.S[0], V26.S4

log_loop4:
    CMP  $4, R2
    BLT  log_scalar_tail
    VLD1.P 16(R1), [V0.S4]
    // V1 = (bits >> 23) - 127  → integer exponent
    VUSHR_S4_BY23(0, 1)
    VISUB_S4(26, 1, 1)
    // V2 = (bits & mantissaMask) | oneBits  → M as f32 bit pattern, M ∈ [1, 2)
    VAND_B16(24, 0, 2)
    VORR_B16(25, 2, 2)
    // Convert exponent to f32.
    VSCVTF_S4(1, 1)
    // y = (M - 1) / (M + 1)
    VFSUB_S4(17, 2, 3)            // V3 = M - 1
    VFADD_S4(17, 2, 4)            // V4 = M + 1
    VFDIV_S4(4, 3, 5)             // V5 = y
    // y² = V6
    VFMUL_S4(5, 5, 6)
    // Polynomial in y² using Horner: p(y²) = c0 + c1*y² + c2*y⁴ + c3*y⁶ + c4*y⁸ + c5*y¹⁰
    // where c_i = 1/(2i+1): c0=1, c1=1/3, c2=1/5, c3=1/7, c4=1/9, c5=1/11.
    // Then log(M) = 2 * y * p(y²).
    VMOV_B16(18, 7)               // p = 1/11
    VMOV_B16(19, 8) ; VFMLA_S4(6, 7, 8)   // p = 1/9 + y²·(1/11)
    VMOV_B16(20, 7) ; VFMLA_S4(6, 8, 7)   // p = 1/7 + y²·...
    VMOV_B16(21, 8) ; VFMLA_S4(6, 7, 8)   // p = 1/5 + y²·...
    VMOV_B16(22, 7) ; VFMLA_S4(6, 8, 7)   // p = 1/3 + y²·...
    VMOV_B16(17, 8) ; VFMLA_S4(6, 7, 8)   // p = 1   + y²·...
    // log(M) = 2 * y * p = V23 * V5 * V8
    VFMUL_S4(5, 8, 8)             // V8 = y * p
    VFMUL_S4(23, 8, 8)            // V8 = 2 * y * p = log(M)
    // result = e * ln2 + log(M)
    VFMLA_S4(16, 1, 8)            // V8 += V16 * V1 → V8 = log(M) + e*ln2
    VST1.P [V8.S4], 16(R0)
    SUB  $4, R2
    B    log_loop4

log_scalar_tail:
    CBZ  R2, log_done

log_scalar_loop:
    FMOVS (R1), F0
    VDUP V0.S[0], V0.S4
    VUSHR_S4_BY23(0, 1)
    VISUB_S4(26, 1, 1)
    VAND_B16(24, 0, 2)
    VORR_B16(25, 2, 2)
    VSCVTF_S4(1, 1)
    VFSUB_S4(17, 2, 3)
    VFADD_S4(17, 2, 4)
    VFDIV_S4(4, 3, 5)
    VFMUL_S4(5, 5, 6)
    VMOV_B16(18, 7)
    VMOV_B16(19, 8) ; VFMLA_S4(6, 7, 8)
    VMOV_B16(20, 7) ; VFMLA_S4(6, 8, 7)
    VMOV_B16(21, 8) ; VFMLA_S4(6, 7, 8)
    VMOV_B16(22, 7) ; VFMLA_S4(6, 8, 7)
    VMOV_B16(17, 8) ; VFMLA_S4(6, 7, 8)
    VFMUL_S4(5, 8, 8)
    VFMUL_S4(23, 8, 8)
    VFMLA_S4(16, 1, 8)
    FMOVS F8, (R0)
    ADD  $4, R1
    ADD  $4, R0
    SUB  $1, R2
    CBNZ R2, log_scalar_loop

log_done:
    RET

#define VFCMGT_S4(m, n, d) WORD $(0x6EA0E400 | ((m) << 16) | ((n) << 5) | (d))
#define VMAX_S4(m, n, d)   WORD $(0x6EA0F400 | ((m) << 16) | ((n) << 5) | (d))
#define VFABS_S4(n, d)     WORD $(0x6EA0F000 | ((n) << 5) | (d))
#define VBSL_B16(m, n, d)  WORD $(0x6E601C00 | ((m) << 16) | ((n) << 5) | (d))

// ReLU follows the element-wise loop shape but holds a zero vector in V8
// across the entire function so the inner loop only pays for FCMGT
// and BSL (no per-iteration VEOR). The scalar reference is:
//   for index, value := range input {
//       out[index] = 0; if value > 0 { out[index] = value }
//   }
// which does NOT propagate NaN. FCMGT returns false on any NaN input,
// so BSL falls through to the zero vector — matches scalar bit-for-bit.
//
// func ReluFloat32NEONAsm(dst, src *float32, n int)
TEXT ·ReLUF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2

    // V8 holds the zero vector across the whole function.
    VEOR V8.B16, V8.B16, V8.B16

relu_loop16:
    CMP  $16, R2
    BLT  relu_loop4
    VLD1.P 64(R1), [V0.S4, V1.S4, V2.S4, V3.S4]
    VFCMGT_S4(8, 0, 4)
    VFCMGT_S4(8, 1, 5)
    VFCMGT_S4(8, 2, 6)
    VFCMGT_S4(8, 3, 7)
    VBSL_B16(8, 0, 4)
    VBSL_B16(8, 1, 5)
    VBSL_B16(8, 2, 6)
    VBSL_B16(8, 3, 7)
    VST1.P [V4.S4, V5.S4, V6.S4, V7.S4], 64(R0)
    SUB  $16, R2
    B    relu_loop16

relu_loop4:
    CMP  $4, R2
    BLT  relu_scalar_tail
    VLD1.P 16(R1), [V0.S4]
    VFCMGT_S4(8, 0, 4)
    VBSL_B16(8, 0, 4)
    VST1.P [V4.S4], 16(R0)
    SUB  $4, R2
    B    relu_loop4

relu_scalar_tail:
    CBZ  R2, relu_done

    // F1 holds scalar zero across the tail.
    FMOVS ZR, F1

relu_scalar_loop:
    FMOVS (R1), F0
    FCMPS F1, F0
    FCSELS GT, F0, F1, F0
    FMOVS F0, (R0)
    ADD  $4, R1
    ADD  $4, R0
    SUB  $1, R2
    CBNZ R2, relu_scalar_loop

relu_done:
    RET

#define VFNEG_S4(n, d)     WORD $(0x6EA0F800 | ((n) << 5) | (d))

DATA  actSigmoidC<>+0(SB)/4, $1.4426950408889634
DATA  actSigmoidC<>+4(SB)/4, $0.6931471805599453
DATA  actSigmoidC<>+8(SB)/4, $127.0
DATA  actSigmoidC<>+12(SB)/4, $0.00019841270
DATA  actSigmoidC<>+16(SB)/4, $0.0013888889
DATA  actSigmoidC<>+20(SB)/4, $0.008333334
DATA  actSigmoidC<>+24(SB)/4, $0.041666667
DATA  actSigmoidC<>+28(SB)/4, $0.16666667
DATA  actSigmoidC<>+32(SB)/4, $0.5
DATA  actSigmoidC<>+36(SB)/4, $1.0
DATA  actSigmoidC<>+40(SB)/4, $1.0
DATA  actSigmoidC<>+44(SB)/4, $2.0
GLOBL actSigmoidC<>(SB), 8, $48

#define LOAD_SIGMOID_CONSTS \
    MOVD $actSigmoidC<>(SB), R3      \
    FMOVS  0(R3), F16                \
    FMOVS  4(R3), F17                \
    FMOVS  8(R3), F18                \
    FMOVS 12(R3), F19                \
    FMOVS 16(R3), F20                \
    FMOVS 20(R3), F21                \
    FMOVS 24(R3), F22                \
    FMOVS 28(R3), F23                \
    FMOVS 32(R3), F24                \
    FMOVS 36(R3), F25                \
    FMOVS 40(R3), F26                \
    VFCVTZS_S4(18, 27)

#define EXP_BODY_SIG(in, out) \
    VFMUL_S4(16, in, 1)              \
    VFRINTN_S4(1, 1)                 \
    VFMUL_S4(17, 1, 2)               \
    VFSUB_S4(2, in, in)              \
    VMOV_B16(19, 3)                  \
    VMOV_B16(20, 4) ; VFMLA_S4(in, 3, 4) \
    VMOV_B16(21, 3) ; VFMLA_S4(in, 4, 3) \
    VMOV_B16(22, 4) ; VFMLA_S4(in, 3, 4) \
    VMOV_B16(23, 3) ; VFMLA_S4(in, 4, 3) \
    VMOV_B16(24, 4) ; VFMLA_S4(in, 3, 4) \
    VMOV_B16(25, 3) ; VFMLA_S4(in, 4, 3) \
    VMOV_B16(26, 4) ; VFMLA_S4(in, 3, 4) \
    VFCVTZS_S4(1, 5)                 \
    VADD_S4(27, 5, 5)                \
    VSHL_S4_BY23(5, 5)               \
    VFMUL_S4(5, 4, out)

// func SigmoidF32NEON(dst, src *float32, count int)
TEXT ·SigmoidF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2
    LOAD_SIGMOID_CONSTS

sigmoid_loop4:
    CMP  $4, R2
    BLT  sigmoid_scalar_tail
    VLD1.P 16(R1), [V0.S4]
    VFNEG_S4(0, 0)
    EXP_BODY_SIG(0, 6)
    VFADD_S4(26, 6, 6)
    VFDIV_S4(6, 26, 7)
    VST1.P [V7.S4], 16(R0)
    SUB  $4, R2
    B    sigmoid_loop4

sigmoid_scalar_tail:
    CBZ  R2, sigmoid_done

sigmoid_scalar_loop:
    FMOVS (R1), F0
    VDUP V0.S[0], V0.S4
    VFNEG_S4(0, 0)
    EXP_BODY_SIG(0, 6)
    VFADD_S4(26, 6, 6)
    VFDIV_S4(6, 26, 7)
    FMOVS F7, (R0)
    ADD  $4, R1
    ADD  $4, R0
    SUB  $1, R2
    CBNZ R2, sigmoid_scalar_loop

sigmoid_done:
    RET

// func SiluF32NEON(dst, src *float32, count int)
TEXT ·SiluF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2
    LOAD_SIGMOID_CONSTS

silu_loop4:
    CMP  $4, R2
    BLT  silu_scalar_tail
    VLD1.P 16(R1), [V0.S4]
    VMOV_B16(0, 8)
    VFNEG_S4(0, 0)
    EXP_BODY_SIG(0, 6)
    VFADD_S4(26, 6, 6)
    VFDIV_S4(6, 8, 7)
    VST1.P [V7.S4], 16(R0)
    SUB  $4, R2
    B    silu_loop4

silu_scalar_tail:
    CBZ  R2, silu_done

silu_scalar_loop:
    FMOVS (R1), F0
    VDUP V0.S[0], V0.S4
    VMOV_B16(0, 8)
    VFNEG_S4(0, 0)
    EXP_BODY_SIG(0, 6)
    VFADD_S4(26, 6, 6)
    VFDIV_S4(6, 8, 7)
    FMOVS F7, (R0)
    ADD  $4, R1
    ADD  $4, R0
    SUB  $1, R2
    CBNZ R2, silu_scalar_loop

silu_done:
    RET

// func TanhF32NEON(dst, src *float32, count int)
TEXT ·TanhF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2
    LOAD_SIGMOID_CONSTS
    FMOVS 44(R3), F28

tanh_loop4:
    CMP  $4, R2
    BLT  tanh_scalar_tail
    VLD1.P 16(R1), [V0.S4]
    VFMUL_S4(28, 0, 0)
    EXP_BODY_SIG(0, 6)
    VFSUB_S4(26, 6, 8)
    VFADD_S4(26, 6, 9)
    VFDIV_S4(9, 8, 7)
    VST1.P [V7.S4], 16(R0)
    SUB  $4, R2
    B    tanh_loop4

tanh_scalar_tail:
    CBZ  R2, tanh_done

tanh_scalar_loop:
    FMOVS (R1), F0
    VDUP V0.S[0], V0.S4
    VFMUL_S4(28, 0, 0)
    EXP_BODY_SIG(0, 6)
    VFSUB_S4(26, 6, 8)
    VFADD_S4(26, 6, 9)
    VFDIV_S4(9, 8, 7)
    FMOVS F7, (R0)
    ADD  $4, R1
    ADD  $4, R0
    SUB  $1, R2
    CBNZ R2, tanh_scalar_loop

tanh_done:
    RET

DATA  actGeluTanhC<>+0(SB)/4, $1.4426950408889634   // log2e
DATA  actGeluTanhC<>+4(SB)/4, $0.6931471805599453   // ln2
DATA  actGeluTanhC<>+8(SB)/4, $127.0                // bias
DATA  actGeluTanhC<>+12(SB)/4, $0.00019841270       // c7
DATA  actGeluTanhC<>+16(SB)/4, $0.0013888889        // c6
DATA  actGeluTanhC<>+20(SB)/4, $0.008333334         // c5
DATA  actGeluTanhC<>+24(SB)/4, $0.041666667         // c4
DATA  actGeluTanhC<>+28(SB)/4, $0.16666667          // c3
DATA  actGeluTanhC<>+32(SB)/4, $0.5                 // c2 / 0.5
DATA  actGeluTanhC<>+36(SB)/4, $1.0                 // c1 / 1
DATA  actGeluTanhC<>+40(SB)/4, $1.0                 // c0
DATA  actGeluTanhC<>+44(SB)/4, $2.0                 // 2.0
DATA  actGeluTanhC<>+48(SB)/4, $0.7978845608028654  // sqrt(2/π)
DATA  actGeluTanhC<>+52(SB)/4, $0.044715            // GELU cubic coefficient
GLOBL actGeluTanhC<>(SB), 8, $56

#define EXP_BODY(in, out) \
    VFMUL_S4(16, in, 1)                  \
    VFRINTN_S4(1, 1)                     \
    VFMUL_S4(17, 1, 2)                   \
    VFSUB_S4(2, in, in)                  \
    VMOV_B16(19, 3)                      \
    VMOV_B16(20, 4) ; VFMLA_S4(in, 3, 4) \
    VMOV_B16(21, 3) ; VFMLA_S4(in, 4, 3) \
    VMOV_B16(22, 4) ; VFMLA_S4(in, 3, 4) \
    VMOV_B16(23, 3) ; VFMLA_S4(in, 4, 3) \
    VMOV_B16(24, 4) ; VFMLA_S4(in, 3, 4) \
    VMOV_B16(25, 3) ; VFMLA_S4(in, 4, 3) \
    VMOV_B16(26, 4) ; VFMLA_S4(in, 3, 4) \
    VFCVTZS_S4(1, 5)                     \
    VADD_S4(27, 5, 5)                    \
    VSHL_S4_BY23(5, 5)                   \
    VFMUL_S4(5, 4, out)

// func GeluTanhFloat32NEONAsm(dst, src *float32, n int)
TEXT ·GeluTanhF32NEON(SB), NOSPLIT, $0-24
    MOVD dst+0(FP), R0
    MOVD src+8(FP), R1
    MOVD n+16(FP), R2

    MOVD $actGeluTanhC<>(SB), R3
    FMOVS  0(R3), F16 ; VDUP V16.S[0], V16.S4
    FMOVS  4(R3), F17 ; VDUP V17.S[0], V17.S4
    FMOVS  8(R3), F18 ; VDUP V18.S[0], V18.S4
    FMOVS 12(R3), F19 ; VDUP V19.S[0], V19.S4
    FMOVS 16(R3), F20 ; VDUP V20.S[0], V20.S4
    FMOVS 20(R3), F21 ; VDUP V21.S[0], V21.S4
    FMOVS 24(R3), F22 ; VDUP V22.S[0], V22.S4
    FMOVS 28(R3), F23 ; VDUP V23.S[0], V23.S4
    FMOVS 32(R3), F24 ; VDUP V24.S[0], V24.S4
    FMOVS 36(R3), F25 ; VDUP V25.S[0], V25.S4
    FMOVS 40(R3), F26 ; VDUP V26.S[0], V26.S4
    VFCVTZS_S4(18, 27)
    FMOVS 44(R3), F28 ; VDUP V28.S[0], V28.S4   // 2.0
    FMOVS 48(R3), F29 ; VDUP V29.S[0], V29.S4   // sqrt(2/π)
    FMOVS 52(R3), F30 ; VDUP V30.S[0], V30.S4   // 0.044715

gelu_loop4:
    CMP  $4, R2
    BLT  gelu_done
    VLD1.P 16(R1), [V0.S4]
    // Stash x in V10 (we need it for the final multiply).
    VMOV_B16(0, 10)
    // V11 = x³ = x * x * x
    VFMUL_S4(0, 0, 11)              // V11 = x²
    VFMUL_S4(11, 0, 11)             // V11 = x³
    // V12 = x + 0.044715 * x³
    VMOV_B16(0, 12)
    VFMLA_S4(11, 30, 12)            // V12 = x + 0.044715*x³ via FMLA
    // V13 = inner = sqrt(2/π) * V12
    VFMUL_S4(29, 12, 13)
    // tanh(inner): compute via e^{2u}-1 / e^{2u}+1
    // 2u into V0 (will be clobbered by EXP_BODY)
    VFMUL_S4(28, 13, 0)             // V0 = 2 * inner
    EXP_BODY(0, 6)                  // V6 = e^{2u}
    VFSUB_S4(26, 6, 14)             // V14 = e^{2u} - 1
    VFADD_S4(26, 6, 15)             // V15 = e^{2u} + 1
    VFDIV_S4(15, 14, 6)             // V6 = tanh(u)
    // gelu = 0.5 * x * (1 + tanh)
    VFADD_S4(26, 6, 6)              // V6 = 1 + tanh
    VFMUL_S4(10, 6, 6)              // V6 = x * (1 + tanh)
    VFMUL_S4(24, 6, 6)              // V6 = 0.5 * V6 (V24 = 0.5)
    VST1.P [V6.S4], 16(R0)
    SUB  $4, R2
    B    gelu_loop4

gelu_done:
    RET

// func SwiGLUTensorsF32NEON(dst, gate, up *float32, count int)
TEXT ·SwiGLUTensorsF32NEON(SB), NOSPLIT, $0-32
    MOVD dst+0(FP), R0
    MOVD gate+8(FP), R1
    MOVD up+16(FP), R2
    MOVD count+24(FP), R3
    LOAD_SIGMOID_CONSTS

swiglu_tensors_w4:
    CMP  $4, R3
    BLT  swiglu_tensors_scalar
    VLD1.P 16(R1), [V0.S4]
    VMOV V0.B16, V1.B16
    VLD1.P 16(R2), [V8.S4]
    VMOV_B16(0, 8)
    VFNEG_S4(0, 0)
    EXP_BODY_SIG(0, 6)
    VFADD_S4(26, 6, 6)
    VFDIV_S4(6, 26, 7)
    VFMUL_S4(1, 7, 7)
    VFMUL_S4(8, 7, 7)
    VST1.P [V7.S4], 16(R0)
    SUB  $4, R3
    B    swiglu_tensors_w4

swiglu_tensors_scalar:
    CBZ  R3, swiglu_tensors_done

swiglu_tensors_sloop:
    FMOVS (R1), F0
    FMOVS (R2), F8
    VDUP V0.S[0], V0.S4
    VMOV V0.B16, V1.B16
    VDUP V8.S[0], V8.S4
    VMOV_B16(0, 8)
    VFNEG_S4(0, 0)
    EXP_BODY_SIG(0, 6)
    VFADD_S4(26, 6, 6)
    VFDIV_S4(6, 26, 7)
    VFMUL_S4(1, 7, 7)
    VFMUL_S4(8, 7, 7)
    FMOVS F7, (R0)
    ADD  $4, R1
    ADD  $4, R2
    ADD  $4, R0
    SUB  $1, R3
    CBNZ R3, swiglu_tensors_sloop

swiglu_tensors_done:
    RET

// func LinGLUTensorsF32NEON(dst, gate, up *float32, count int)
TEXT ·LinGLUTensorsF32NEON(SB), NOSPLIT, $0-32
    MOVD dst+0(FP), R0
    MOVD gate+8(FP), R1
    MOVD up+16(FP), R2
    MOVD count+24(FP), R3

linglu_tensors_w4:
    CMP  $4, R3
    BLT  linglu_tensors_scalar
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V8.S4]
    VFMUL_S4(0, 8, 7)
    VST1.P [V7.S4], 16(R0)
    SUB  $4, R3
    B    linglu_tensors_w4

linglu_tensors_scalar:
    CBZ  R3, linglu_tensors_done

linglu_tensors_sloop:
    FMOVS (R1), F0
    FMOVS (R2), F8
    VDUP V0.S[0], V0.S4
    VDUP V8.S[0], V8.S4
    VFMUL_S4(0, 8, 7)
    FMOVS F7, (R0)
    ADD  $4, R1
    ADD  $4, R2
    ADD  $4, R0
    SUB  $1, R3
    CBNZ R3, linglu_tensors_sloop

linglu_tensors_done:
    RET

// func ReGLUTensorsF32NEON(dst, gate, up *float32, count int)
TEXT ·ReGLUTensorsF32NEON(SB), NOSPLIT, $0-32
    MOVD dst+0(FP), R0
    MOVD gate+8(FP), R1
    MOVD up+16(FP), R2
    MOVD count+24(FP), R3
    VEOR V9.B16, V9.B16, V9.B16

reglu_tensors_w4:
    CMP  $4, R3
    BLT  reglu_tensors_scalar
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V8.S4]
    VMAX_S4(9, 8, 7)
    VFMUL_S4(0, 7, 7)
    VST1.P [V7.S4], 16(R0)
    SUB  $4, R3
    B    reglu_tensors_w4

reglu_tensors_scalar:
    CBZ  R3, reglu_tensors_done

reglu_tensors_sloop:
    FMOVS (R1), F0
    FMOVS (R2), F8
    VDUP V0.S[0], V0.S4
    VDUP V8.S[0], V8.S4
    VMAX_S4(9, 8, 7)
    VFMUL_S4(0, 7, 7)
    FMOVS F7, (R0)
    ADD  $4, R1
    ADD  $4, R2
    ADD  $4, R0
    SUB  $1, R3
    CBNZ R3, reglu_tensors_sloop

reglu_tensors_done:
    RET

// func GLUTensorsF32NEON(dst, gate, up *float32, count int)
TEXT ·GLUTensorsF32NEON(SB), NOSPLIT, $0-32
    MOVD dst+0(FP), R0
    MOVD gate+8(FP), R1
    MOVD up+16(FP), R2
    MOVD count+24(FP), R3
    LOAD_SIGMOID_CONSTS

glu_tensors_w4:
    CMP  $4, R3
    BLT  glu_tensors_scalar
    VLD1.P 16(R1), [V10.S4]
    VLD1.P 16(R2), [V0.S4]
    VFNEG_S4(0, 0)
    EXP_BODY_SIG(0, 6)
    VFADD_S4(26, 6, 6)
    VFDIV_S4(6, 26, 7)
    VFMUL_S4(10, 7, 7)
    VST1.P [V7.S4], 16(R0)
    SUB  $4, R3
    B    glu_tensors_w4

glu_tensors_scalar:
    CBZ  R3, glu_tensors_done

glu_tensors_sloop:
    FMOVS (R1), F10
    FMOVS (R2), F0
    VDUP V10.S[0], V10.S4
    VDUP V0.S[0], V0.S4
    VFNEG_S4(0, 0)
    EXP_BODY_SIG(0, 6)
    VFADD_S4(26, 6, 6)
    VFDIV_S4(6, 26, 7)
    VFMUL_S4(10, 7, 7)
    FMOVS F7, (R0)
    ADD  $4, R1
    ADD  $4, R2
    ADD  $4, R0
    SUB  $1, R3
    CBNZ R3, glu_tensors_sloop

glu_tensors_done:
    RET

// func SiGLUTensorsF32NEON(dst, gate, up *float32, count int)
TEXT ·SiGLUTensorsF32NEON(SB), NOSPLIT, $0-32
    MOVD dst+0(FP), R0
    MOVD gate+8(FP), R1
    MOVD up+16(FP), R2
    MOVD count+24(FP), R3
    LOAD_SIGMOID_CONSTS

siglu_tensors_w4:
    CMP  $4, R3
    BLT  siglu_tensors_scalar
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V8.S4]
    VFNEG_S4(0, 0)
    EXP_BODY_SIG(0, 6)
    VFADD_S4(26, 6, 6)
    VFDIV_S4(6, 26, 7)
    VFMUL_S4(8, 7, 7)
    VST1.P [V7.S4], 16(R0)
    SUB  $4, R3
    B    siglu_tensors_w4

siglu_tensors_scalar:
    CBZ  R3, siglu_tensors_done

siglu_tensors_sloop:
    FMOVS (R1), F0
    FMOVS (R2), F8
    VDUP V0.S[0], V0.S4
    VDUP V8.S[0], V8.S4
    VFNEG_S4(0, 0)
    EXP_BODY_SIG(0, 6)
    VFADD_S4(26, 6, 6)
    VFDIV_S4(6, 26, 7)
    VFMUL_S4(8, 7, 7)
    FMOVS F7, (R0)
    ADD  $4, R1
    ADD  $4, R2
    ADD  $4, R0
    SUB  $1, R3
    CBNZ R3, siglu_tensors_sloop

siglu_tensors_done:
    RET

// func SeGLUTensorsF32NEON(dst, gate, up *float32, count int)
TEXT ·SeGLUTensorsF32NEON(SB), NOSPLIT, $0-32
    MOVD dst+0(FP), R0
    MOVD gate+8(FP), R1
    MOVD up+16(FP), R2
    MOVD count+24(FP), R3
    LOAD_SIGMOID_CONSTS

seglu_tensors_w4:
    CMP  $4, R3
    BLT  seglu_tensors_scalar
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V8.S4]
    VFNEG_S4(0, 0)
    EXP_BODY_SIG(0, 6)
    VFADD_S4(26, 6, 6)
    VFDIV_S4(6, 26, 7)
    VFMUL_S4(8, 7, 7)
    VST1.P [V7.S4], 16(R0)
    SUB  $4, R3
    B    seglu_tensors_w4

seglu_tensors_scalar:
    CBZ  R3, seglu_tensors_done

seglu_tensors_sloop:
    FMOVS (R1), F0
    FMOVS (R2), F8
    VDUP V0.S[0], V0.S4
    VDUP V8.S[0], V8.S4
    VFNEG_S4(0, 0)
    EXP_BODY_SIG(0, 6)
    VFADD_S4(26, 6, 6)
    VFDIV_S4(6, 26, 7)
    VFMUL_S4(8, 7, 7)
    FMOVS F7, (R0)
    ADD  $4, R1
    ADD  $4, R2
    ADD  $4, R0
    SUB  $1, R3
    CBNZ R3, seglu_tensors_sloop

seglu_tensors_done:
    RET

#define GELU_ERF_ON_UP_NEON \
    VMOV_B16(0, 10) ;\
    FMOVS 36(R4), F29 ;\
    VFMUL_S4(29, 0, 11) ;\
    VFABS_S4(11, 12) ;\
    FMOVS 0(R4), F30 ;\
    VFADD_S4(30, 12, 13) ;\
    VFDIV_S4(13, 12, 12) ;\
    VFNEG_S4(11, 14) ;\
    VFMUL_S4(14, 14, 15) ;\
    VFMLA_S4(11, 12, 14) ;\
    VFMUL_S4(14, 12, 14) ;\
    VFSUB_S4(26, 14, 14) ;\
    VFCMGT_S4(8, 11, 13) ;\
    VFNEG_S4(14, 15) ;\
    VBSL_B16(13, 15, 14) ;\
    VFADD_S4(26, 14, 14) ;\
    VFMUL_S4(10, 14, 7) ;\
    VFMUL_S4(24, 7, 7)

#define GELU_TANH_ON_UP_NEON \
    VMOV_B16(0, 10) ;\
    VFMUL_S4(0, 0, 11) ;\
    VFMUL_S4(11, 0, 11) ;\
    VMOV_B16(0, 12) ;\
    VFMLA_S4(11, 30, 12) ;\
    VFMUL_S4(29, 12, 13) ;\
    VFMUL_S4(28, 13, 0) ;\
    EXP_BODY(0, 6) ;\
    VFSUB_S4(26, 6, 14) ;\
    VFADD_S4(26, 6, 15) ;\
    VFDIV_S4(15, 14, 6) ;\
    VFADD_S4(26, 6, 6) ;\
    VFMUL_S4(10, 6, 6) ;\
    VFMUL_S4(24, 6, 7)

// func GeGLUTensorsF32NEON(dst, gate, up *float32, count int)
TEXT ·GeGLUTensorsF32NEON(SB), NOSPLIT, $0-32
    MOVD dst+0(FP), R0
    MOVD gate+8(FP), R1
    MOVD up+16(FP), R2
    MOVD count+24(FP), R3
    MOVD $actExtraMiscC<>(SB), R4
    MOVD $actExpC<>(SB), R5
    FMOVS 40(R5), F26
    FMOVS 32(R4), F24
    VEOR V8.B16, V8.B16, V8.B16

geglu_tensors_w4:
    CMP  $4, R3
    BLT  geglu_tensors_scalar
    VLD1.P 16(R1), [V3.S4]
    VLD1.P 16(R2), [V0.S4]
    GELU_ERF_ON_UP_NEON
    VFMUL_S4(3, 7, 7)
    VST1.P [V7.S4], 16(R0)
    SUB  $4, R3
    B    geglu_tensors_w4

geglu_tensors_scalar:
    CBZ  R3, geglu_tensors_done

geglu_tensors_sloop:
    FMOVS (R1), F3
    FMOVS (R2), F0
    VDUP V3.S[0], V3.S4
    VDUP V0.S[0], V0.S4
    GELU_ERF_ON_UP_NEON
    VFMUL_S4(3, 7, 7)
    FMOVS F7, (R0)
    ADD  $4, R1
    ADD  $4, R2
    ADD  $4, R0
    SUB  $1, R3
    CBNZ R3, geglu_tensors_sloop

geglu_tensors_done:
    RET

// func GeGLUTanhTensorsF32NEON(dst, gate, up *float32, count int)
TEXT ·GeGLUTanhTensorsF32NEON(SB), NOSPLIT, $0-32
    MOVD dst+0(FP), R0
    MOVD gate+8(FP), R1
    MOVD up+16(FP), R2
    MOVD count+24(FP), R3
    MOVD $actGeluTanhC<>(SB), R4
    FMOVS  0(R4), F16
    FMOVS  4(R4), F17
    FMOVS  8(R4), F18
    FMOVS 12(R4), F19
    FMOVS 16(R4), F20
    FMOVS 20(R4), F21
    FMOVS 24(R4), F22
    FMOVS 28(R4), F23
    FMOVS 32(R4), F24
    FMOVS 36(R4), F25
    FMOVS 40(R4), F26
    VFCVTZS_S4(18, 27)
    FMOVS 44(R4), F28
    FMOVS 48(R4), F29
    FMOVS 52(R4), F30

geglutanh_tensors_w4:
    CMP  $4, R3
    BLT  geglutanh_tensors_scalar
    VLD1.P 16(R1), [V3.S4]
    VLD1.P 16(R2), [V0.S4]
    GELU_TANH_ON_UP_NEON
    VFMUL_S4(3, 7, 7)
    VST1.P [V7.S4], 16(R0)
    SUB  $4, R3
    B    geglutanh_tensors_w4

geglutanh_tensors_scalar:
    CBZ  R3, geglutanh_tensors_done

geglutanh_tensors_sloop:
    FMOVS (R1), F3
    FMOVS (R2), F0
    VDUP V3.S[0], V3.S4
    VDUP V0.S[0], V0.S4
    GELU_TANH_ON_UP_NEON
    VFMUL_S4(3, 7, 7)
    FMOVS F7, (R0)
    ADD  $4, R1
    ADD  $4, R2
    ADD  $4, R0
    SUB  $1, R3
    CBNZ R3, geglutanh_tensors_sloop

geglutanh_tensors_done:
    RET
