// SPDX-License-Identifier: Apache-2.0
// AVX2 packed gate+up layout: [gate₀…gateₙ₋₁][up₀…upₙ₋₁] per batch row.
#include "textflag.h"

// func SwiGLUPackedF32AVX2(dst, packed *float32, batch, halfCount int)
TEXT ·SwiGLUPackedF32AVX2(SB), NOSPLIT, $32-28
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ swiglu_packed_done
	MOVQ R9, R10
	SHLQ $2, R10
swiglu_packed_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·SwiGLUTensorsF32AVX2(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ swiglu_packed_row
swiglu_packed_done:
	RET

// func LinGLUPackedF32AVX2(dst, packed *float32, batch, halfCount int)
TEXT ·LinGLUPackedF32AVX2(SB), NOSPLIT, $32-28
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ linglu_packed_done
	MOVQ R9, R10
	SHLQ $2, R10
linglu_packed_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·LinGLUTensorsF32AVX2(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ linglu_packed_row
linglu_packed_done:
	RET

// func ReGLUPackedF32AVX2(dst, packed *float32, batch, halfCount int)
TEXT ·ReGLUPackedF32AVX2(SB), NOSPLIT, $32-28
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ reglu_packed_done
	MOVQ R9, R10
	SHLQ $2, R10
reglu_packed_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·ReGLUTensorsF32AVX2(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ reglu_packed_row
reglu_packed_done:
	RET

// func GLUPackedF32AVX2(dst, packed *float32, batch, halfCount int)
TEXT ·GLUPackedF32AVX2(SB), NOSPLIT, $32-28
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ glu_packed_done
	MOVQ R9, R10
	SHLQ $2, R10
glu_packed_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·GLUTensorsF32AVX2(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ glu_packed_row
glu_packed_done:
	RET

// func SiGLUPackedF32AVX2(dst, packed *float32, batch, halfCount int)
TEXT ·SiGLUPackedF32AVX2(SB), NOSPLIT, $32-28
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ siglu_packed_done
	MOVQ R9, R10
	SHLQ $2, R10
siglu_packed_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·SiGLUTensorsF32AVX2(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ siglu_packed_row
siglu_packed_done:
	RET

// func SeGLUPackedF32AVX2(dst, packed *float32, batch, halfCount int)
TEXT ·SeGLUPackedF32AVX2(SB), NOSPLIT, $32-28
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ seglu_packed_done
	MOVQ R9, R10
	SHLQ $2, R10
seglu_packed_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·SeGLUTensorsF32AVX2(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ seglu_packed_row
seglu_packed_done:
	RET

// func GeGLUPackedF32AVX2(dst, packed *float32, batch, halfCount int)
TEXT ·GeGLUPackedF32AVX2(SB), NOSPLIT, $32-28
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ geglu_packed_done
	MOVQ R9, R10
	SHLQ $2, R10
geglu_packed_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·GeGLUTensorsF32AVX2(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ geglu_packed_row
geglu_packed_done:
	RET

// func GeGLUTanhPackedF32AVX2(dst, packed *float32, batch, halfCount int)
TEXT ·GeGLUTanhPackedF32AVX2(SB), NOSPLIT, $32-28
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ geglu_tanh_packed_done
	MOVQ R9, R10
	SHLQ $2, R10
geglu_tanh_packed_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·GeGLUTanhTensorsF32AVX2(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ geglu_tanh_packed_row
geglu_tanh_packed_done:
	RET
