// SPDX-License-Identifier: Apache-2.0
// 386 SSE2 packed gate+up layout.
#include "textflag.h"

// func SwiGLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
TEXT ·SwiGLUPackedF32SSE2(SB), NOSPLIT, $32-20
	MOVL dst+0(FP), DI
	MOVL packed+4(FP), SI
	MOVL batch+8(FP), DX
	MOVL halfCount+12(FP), CX
	MOVL CX, 24(SP)
	MOVL CX, AX
	SHLL $2, AX
	MOVL AX, 28(SP)
swiglu_packed_sse2_386_row:
	TESTL DX, DX
	JZ swiglu_packed_sse2_386_done
	MOVL DI, 0(SP)
	MOVL SI, 4(SP)
	MOVL SI, BX
	MOVL 28(SP), AX
	ADDL AX, BX
	MOVL BX, 8(SP)
	MOVL 24(SP), AX
	MOVL AX, 12(SP)
	CALL ·SwiGLUTensorsF32SSE2(SB)
	MOVL 24(SP), AX
	SHLL $3, AX
	ADDL AX, SI
	MOVL 24(SP), AX
	SHLL $2, AX
	ADDL AX, DI
	DECL DX
	JMP swiglu_packed_sse2_386_row
swiglu_packed_sse2_386_done:
	RET

// func LinGLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
TEXT ·LinGLUPackedF32SSE2(SB), NOSPLIT, $32-20
	MOVL dst+0(FP), DI
	MOVL packed+4(FP), SI
	MOVL batch+8(FP), DX
	MOVL halfCount+12(FP), CX
	MOVL CX, 24(SP)
	MOVL CX, AX
	SHLL $2, AX
	MOVL AX, 28(SP)
linglu_packed_sse2_386_row:
	TESTL DX, DX
	JZ linglu_packed_sse2_386_done
	MOVL DI, 0(SP)
	MOVL SI, 4(SP)
	MOVL SI, BX
	MOVL 28(SP), AX
	ADDL AX, BX
	MOVL BX, 8(SP)
	MOVL 24(SP), AX
	MOVL AX, 12(SP)
	CALL ·LinGLUTensorsF32SSE2(SB)
	MOVL 24(SP), AX
	SHLL $3, AX
	ADDL AX, SI
	MOVL 24(SP), AX
	SHLL $2, AX
	ADDL AX, DI
	DECL DX
	JMP linglu_packed_sse2_386_row
linglu_packed_sse2_386_done:
	RET

// func ReGLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
TEXT ·ReGLUPackedF32SSE2(SB), NOSPLIT, $32-20
	MOVL dst+0(FP), DI
	MOVL packed+4(FP), SI
	MOVL batch+8(FP), DX
	MOVL halfCount+12(FP), CX
	MOVL CX, 24(SP)
	MOVL CX, AX
	SHLL $2, AX
	MOVL AX, 28(SP)
reglu_packed_sse2_386_row:
	TESTL DX, DX
	JZ reglu_packed_sse2_386_done
	MOVL DI, 0(SP)
	MOVL SI, 4(SP)
	MOVL SI, BX
	MOVL 28(SP), AX
	ADDL AX, BX
	MOVL BX, 8(SP)
	MOVL 24(SP), AX
	MOVL AX, 12(SP)
	CALL ·ReGLUTensorsF32SSE2(SB)
	MOVL 24(SP), AX
	SHLL $3, AX
	ADDL AX, SI
	MOVL 24(SP), AX
	SHLL $2, AX
	ADDL AX, DI
	DECL DX
	JMP reglu_packed_sse2_386_row
reglu_packed_sse2_386_done:
	RET

// func GLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
TEXT ·GLUPackedF32SSE2(SB), NOSPLIT, $32-20
	MOVL dst+0(FP), DI
	MOVL packed+4(FP), SI
	MOVL batch+8(FP), DX
	MOVL halfCount+12(FP), CX
	MOVL CX, 24(SP)
	MOVL CX, AX
	SHLL $2, AX
	MOVL AX, 28(SP)
glu_packed_sse2_386_row:
	TESTL DX, DX
	JZ glu_packed_sse2_386_done
	MOVL DI, 0(SP)
	MOVL SI, 4(SP)
	MOVL SI, BX
	MOVL 28(SP), AX
	ADDL AX, BX
	MOVL BX, 8(SP)
	MOVL 24(SP), AX
	MOVL AX, 12(SP)
	CALL ·GLUTensorsF32SSE2(SB)
	MOVL 24(SP), AX
	SHLL $3, AX
	ADDL AX, SI
	MOVL 24(SP), AX
	SHLL $2, AX
	ADDL AX, DI
	DECL DX
	JMP glu_packed_sse2_386_row
glu_packed_sse2_386_done:
	RET

// func SiGLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
TEXT ·SiGLUPackedF32SSE2(SB), NOSPLIT, $32-20
	MOVL dst+0(FP), DI
	MOVL packed+4(FP), SI
	MOVL batch+8(FP), DX
	MOVL halfCount+12(FP), CX
	MOVL CX, 24(SP)
	MOVL CX, AX
	SHLL $2, AX
	MOVL AX, 28(SP)
siglu_packed_sse2_386_row:
	TESTL DX, DX
	JZ siglu_packed_sse2_386_done
	MOVL DI, 0(SP)
	MOVL SI, 4(SP)
	MOVL SI, BX
	MOVL 28(SP), AX
	ADDL AX, BX
	MOVL BX, 8(SP)
	MOVL 24(SP), AX
	MOVL AX, 12(SP)
	CALL ·SiGLUTensorsF32SSE2(SB)
	MOVL 24(SP), AX
	SHLL $3, AX
	ADDL AX, SI
	MOVL 24(SP), AX
	SHLL $2, AX
	ADDL AX, DI
	DECL DX
	JMP siglu_packed_sse2_386_row
siglu_packed_sse2_386_done:
	RET

// func SeGLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
TEXT ·SeGLUPackedF32SSE2(SB), NOSPLIT, $32-20
	MOVL dst+0(FP), DI
	MOVL packed+4(FP), SI
	MOVL batch+8(FP), DX
	MOVL halfCount+12(FP), CX
	MOVL CX, 24(SP)
	MOVL CX, AX
	SHLL $2, AX
	MOVL AX, 28(SP)
seglu_packed_sse2_386_row:
	TESTL DX, DX
	JZ seglu_packed_sse2_386_done
	MOVL DI, 0(SP)
	MOVL SI, 4(SP)
	MOVL SI, BX
	MOVL 28(SP), AX
	ADDL AX, BX
	MOVL BX, 8(SP)
	MOVL 24(SP), AX
	MOVL AX, 12(SP)
	CALL ·SeGLUTensorsF32SSE2(SB)
	MOVL 24(SP), AX
	SHLL $3, AX
	ADDL AX, SI
	MOVL 24(SP), AX
	SHLL $2, AX
	ADDL AX, DI
	DECL DX
	JMP seglu_packed_sse2_386_row
seglu_packed_sse2_386_done:
	RET

// func GeGLUPackedF32SSE2(dst, packed *float32, batch, halfCount int)
TEXT ·GeGLUPackedF32SSE2(SB), NOSPLIT, $32-20
	MOVL dst+0(FP), DI
	MOVL packed+4(FP), SI
	MOVL batch+8(FP), DX
	MOVL halfCount+12(FP), CX
	MOVL CX, 24(SP)
	MOVL CX, AX
	SHLL $2, AX
	MOVL AX, 28(SP)
geglu_packed_sse2_386_row:
	TESTL DX, DX
	JZ geglu_packed_sse2_386_done
	MOVL DI, 0(SP)
	MOVL SI, 4(SP)
	MOVL SI, BX
	MOVL 28(SP), AX
	ADDL AX, BX
	MOVL BX, 8(SP)
	MOVL 24(SP), AX
	MOVL AX, 12(SP)
	CALL ·GeGLUTensorsF32SSE2(SB)
	MOVL 24(SP), AX
	SHLL $3, AX
	ADDL AX, SI
	MOVL 24(SP), AX
	SHLL $2, AX
	ADDL AX, DI
	DECL DX
	JMP geglu_packed_sse2_386_row
geglu_packed_sse2_386_done:
	RET

// func GeGLUTanhPackedF32SSE2(dst, packed *float32, batch, halfCount int)
TEXT ·GeGLUTanhPackedF32SSE2(SB), NOSPLIT, $32-20
	MOVL dst+0(FP), DI
	MOVL packed+4(FP), SI
	MOVL batch+8(FP), DX
	MOVL halfCount+12(FP), CX
	MOVL CX, 24(SP)
	MOVL CX, AX
	SHLL $2, AX
	MOVL AX, 28(SP)
geglu_tanh_packed_sse2_386_row:
	TESTL DX, DX
	JZ geglu_tanh_packed_sse2_386_done
	MOVL DI, 0(SP)
	MOVL SI, 4(SP)
	MOVL SI, BX
	MOVL 28(SP), AX
	ADDL AX, BX
	MOVL BX, 8(SP)
	MOVL 24(SP), AX
	MOVL AX, 12(SP)
	CALL ·GeGLUTanhTensorsF32SSE2(SB)
	MOVL 24(SP), AX
	SHLL $3, AX
	ADDL AX, SI
	MOVL 24(SP), AX
	SHLL $2, AX
	ADDL AX, DI
	DECL DX
	JMP geglu_tanh_packed_sse2_386_row
geglu_tanh_packed_sse2_386_done:
	RET
