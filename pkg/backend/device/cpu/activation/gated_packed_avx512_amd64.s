// SPDX-License-Identifier: Apache-2.0
// AVX-512 packed gate+up layout.
#include "textflag.h"

// func SwiGLUPackedF32AVX512(dst, packed *float32, batch, halfCount int)
TEXT ·SwiGLUPackedF32AVX512(SB), NOSPLIT, $32-32
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ swiglu_packed_avx512_done
	MOVQ R9, R10
	SHLQ $2, R10
swiglu_packed_avx512_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·SwiGLUTensorsF32AVX512(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ swiglu_packed_avx512_row
swiglu_packed_avx512_done:
	RET

// func LinGLUPackedF32AVX512(dst, packed *float32, batch, halfCount int)
TEXT ·LinGLUPackedF32AVX512(SB), NOSPLIT, $32-32
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ linglu_packed_avx512_done
	MOVQ R9, R10
	SHLQ $2, R10
linglu_packed_avx512_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·LinGLUTensorsF32AVX512(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ linglu_packed_avx512_row
linglu_packed_avx512_done:
	RET

// func ReGLUPackedF32AVX512(dst, packed *float32, batch, halfCount int)
TEXT ·ReGLUPackedF32AVX512(SB), NOSPLIT, $32-32
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ reglu_packed_avx512_done
	MOVQ R9, R10
	SHLQ $2, R10
reglu_packed_avx512_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·ReGLUTensorsF32AVX512(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ reglu_packed_avx512_row
reglu_packed_avx512_done:
	RET

// func GLUPackedF32AVX512(dst, packed *float32, batch, halfCount int)
TEXT ·GLUPackedF32AVX512(SB), NOSPLIT, $32-32
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ glu_packed_avx512_done
	MOVQ R9, R10
	SHLQ $2, R10
glu_packed_avx512_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·GLUTensorsF32AVX512(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ glu_packed_avx512_row
glu_packed_avx512_done:
	RET

// func SiGLUPackedF32AVX512(dst, packed *float32, batch, halfCount int)
TEXT ·SiGLUPackedF32AVX512(SB), NOSPLIT, $32-32
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ siglu_packed_avx512_done
	MOVQ R9, R10
	SHLQ $2, R10
siglu_packed_avx512_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·SiGLUTensorsF32AVX512(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ siglu_packed_avx512_row
siglu_packed_avx512_done:
	RET

// func SeGLUPackedF32AVX512(dst, packed *float32, batch, halfCount int)
TEXT ·SeGLUPackedF32AVX512(SB), NOSPLIT, $32-32
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ seglu_packed_avx512_done
	MOVQ R9, R10
	SHLQ $2, R10
seglu_packed_avx512_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·SeGLUTensorsF32AVX512(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ seglu_packed_avx512_row
seglu_packed_avx512_done:
	RET

// func GeGLUPackedF32AVX512(dst, packed *float32, batch, halfCount int)
TEXT ·GeGLUPackedF32AVX512(SB), NOSPLIT, $32-32
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ geglu_packed_avx512_done
	MOVQ R9, R10
	SHLQ $2, R10
geglu_packed_avx512_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·GeGLUTensorsF32AVX512(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ geglu_packed_avx512_row
geglu_packed_avx512_done:
	RET

// func GeGLUTanhPackedF32AVX512(dst, packed *float32, batch, halfCount int)
TEXT ·GeGLUTanhPackedF32AVX512(SB), NOSPLIT, $32-32
	MOVQ dst+0(FP), DI
	MOVQ packed+8(FP), SI
	MOVQ batch+16(FP), BX
	MOVQ halfCount+24(FP), R9
	TESTQ BX, BX
	JZ geglu_tanh_packed_avx512_done
	MOVQ R9, R10
	SHLQ $2, R10
geglu_tanh_packed_avx512_row:
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ SI, R11
	ADDQ R10, R11
	MOVQ R11, 16(SP)
	MOVQ R9, 24(SP)
	CALL ·GeGLUTanhTensorsF32AVX512(SB)
	MOVQ R9, AX
	SHLQ $3, AX
	ADDQ AX, SI
	MOVQ R9, AX
	SHLQ $2, AX
	ADDQ AX, DI
	DECQ BX
	JNZ geglu_tanh_packed_avx512_row
geglu_tanh_packed_avx512_done:
	RET
