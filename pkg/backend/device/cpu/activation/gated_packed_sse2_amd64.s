// SPDX-License-Identifier: Apache-2.0
// SSE2 packed gate+up layout.
#include "textflag.h"

// func SwiGLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
TEXT ·SwiGLUPackedF32SSE2(SB), NOSPLIT, $32-28
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ swiglu_packed_sse2_done
	MOVQ R9, R10
	SHLQ $2, R10
swiglu_packed_sse2_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·SwiGLUTensorsF32SSE2(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ swiglu_packed_sse2_row
swiglu_packed_sse2_done:
	RET

// func LinGLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
TEXT ·LinGLUPackedF32SSE2(SB), NOSPLIT, $32-28
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ linglu_packed_sse2_done
	MOVQ R9, R10
	SHLQ $2, R10
linglu_packed_sse2_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·LinGLUTensorsF32SSE2(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ linglu_packed_sse2_row
linglu_packed_sse2_done:
	RET

// func ReGLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
TEXT ·ReGLUPackedF32SSE2(SB), NOSPLIT, $32-28
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ reglu_packed_sse2_done
	MOVQ R9, R10
	SHLQ $2, R10
reglu_packed_sse2_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·ReGLUTensorsF32SSE2(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ reglu_packed_sse2_row
reglu_packed_sse2_done:
	RET

// func GLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
TEXT ·GLUPackedF32SSE2(SB), NOSPLIT, $32-28
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ glu_packed_sse2_done
	MOVQ R9, R10
	SHLQ $2, R10
glu_packed_sse2_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·GLUTensorsF32SSE2(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ glu_packed_sse2_row
glu_packed_sse2_done:
	RET

// func SiGLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
TEXT ·SiGLUPackedF32SSE2(SB), NOSPLIT, $32-28
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ siglu_packed_sse2_done
	MOVQ R9, R10
	SHLQ $2, R10
siglu_packed_sse2_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·SiGLUTensorsF32SSE2(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ siglu_packed_sse2_row
siglu_packed_sse2_done:
	RET

// func SeGLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
TEXT ·SeGLUPackedF32SSE2(SB), NOSPLIT, $32-28
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ seglu_packed_sse2_done
	MOVQ R9, R10
	SHLQ $2, R10
seglu_packed_sse2_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·SeGLUTensorsF32SSE2(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ seglu_packed_sse2_row
seglu_packed_sse2_done:
	RET

// func GeGLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
TEXT ·GeGLUPackedF32SSE2(SB), NOSPLIT, $32-28
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ geglu_packed_sse2_done
	MOVQ R9, R10
	SHLQ $2, R10
geglu_packed_sse2_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·GeGLUTensorsF32SSE2(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ geglu_packed_sse2_row
geglu_packed_sse2_done:
	RET

// func GeGLUTanhPackedF32SSE2(dst, packed *float32, batch, halfCount int)
TEXT ·GeGLUTanhPackedF32SSE2(SB), NOSPLIT, $32-28
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ geglu_tanh_packed_sse2_done
	MOVQ R9, R10
	SHLQ $2, R10
geglu_tanh_packed_sse2_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·GeGLUTanhTensorsF32SSE2(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ geglu_tanh_packed_sse2_row
geglu_tanh_packed_sse2_done:
	RET
