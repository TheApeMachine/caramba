// SPDX-License-Identifier: Apache-2.0
// NEON packed gate+up layout.
#include "textflag.h"

// func SwiGLUPackedF32NEON(dst, packed *float32, batch, halfCount int)
TEXT ·SwiGLUPackedF32NEON(SB), NOSPLIT, $32-32
	MOVD dst+0(FP), R4
	MOVD packed+8(FP), R5
	MOVD batch+16(FP), R6
	MOVD halfCount+24(FP), R7
	CBZ R6, swiglu_packed_neon_done
	LSL $2, R7, R8
swiglu_packed_neon_row:
	MOVD R4, 0(RSP)
	MOVD R5, 8(RSP)
	ADD R8, R5, R9
	MOVD R9, 16(RSP)
	MOVD R7, 24(RSP)
	CALL ·SwiGLUTensorsF32NEON(SB)
	LSL $3, R7, R10
	ADD R10, R5
	LSL $2, R7, R10
	ADD R10, R4
	SUB $1, R6
	CBNZ R6, swiglu_packed_neon_row
swiglu_packed_neon_done:
	RET

// func LinGLUPackedF32NEON(dst, packed *float32, batch, halfCount int)
TEXT ·LinGLUPackedF32NEON(SB), NOSPLIT, $32-32
	MOVD dst+0(FP), R4
	MOVD packed+8(FP), R5
	MOVD batch+16(FP), R6
	MOVD halfCount+24(FP), R7
	CBZ R6, linglu_packed_neon_done
	LSL $2, R7, R8
linglu_packed_neon_row:
	MOVD R4, 0(RSP)
	MOVD R5, 8(RSP)
	ADD R8, R5, R9
	MOVD R9, 16(RSP)
	MOVD R7, 24(RSP)
	CALL ·LinGLUTensorsF32NEON(SB)
	LSL $3, R7, R10
	ADD R10, R5
	LSL $2, R7, R10
	ADD R10, R4
	SUB $1, R6
	CBNZ R6, linglu_packed_neon_row
linglu_packed_neon_done:
	RET

// func ReGLUPackedF32NEON(dst, packed *float32, batch, halfCount int)
TEXT ·ReGLUPackedF32NEON(SB), NOSPLIT, $32-32
	MOVD dst+0(FP), R4
	MOVD packed+8(FP), R5
	MOVD batch+16(FP), R6
	MOVD halfCount+24(FP), R7
	CBZ R6, reglu_packed_neon_done
	LSL $2, R7, R8
reglu_packed_neon_row:
	MOVD R4, 0(RSP)
	MOVD R5, 8(RSP)
	ADD R8, R5, R9
	MOVD R9, 16(RSP)
	MOVD R7, 24(RSP)
	CALL ·ReGLUTensorsF32NEON(SB)
	LSL $3, R7, R10
	ADD R10, R5
	LSL $2, R7, R10
	ADD R10, R4
	SUB $1, R6
	CBNZ R6, reglu_packed_neon_row
reglu_packed_neon_done:
	RET

// func GLUPackedF32NEON(dst, packed *float32, batch, halfCount int)
TEXT ·GLUPackedF32NEON(SB), NOSPLIT, $32-32
	MOVD dst+0(FP), R4
	MOVD packed+8(FP), R5
	MOVD batch+16(FP), R6
	MOVD halfCount+24(FP), R7
	CBZ R6, glu_packed_neon_done
	LSL $2, R7, R8
glu_packed_neon_row:
	MOVD R4, 0(RSP)
	MOVD R5, 8(RSP)
	ADD R8, R5, R9
	MOVD R9, 16(RSP)
	MOVD R7, 24(RSP)
	CALL ·GLUTensorsF32NEON(SB)
	LSL $3, R7, R10
	ADD R10, R5
	LSL $2, R7, R10
	ADD R10, R4
	SUB $1, R6
	CBNZ R6, glu_packed_neon_row
glu_packed_neon_done:
	RET

// func SiGLUPackedF32NEON(dst, packed *float32, batch, halfCount int)
TEXT ·SiGLUPackedF32NEON(SB), NOSPLIT, $32-32
	MOVD dst+0(FP), R4
	MOVD packed+8(FP), R5
	MOVD batch+16(FP), R6
	MOVD halfCount+24(FP), R7
	CBZ R6, siglu_packed_neon_done
	LSL $2, R7, R8
siglu_packed_neon_row:
	MOVD R4, 0(RSP)
	MOVD R5, 8(RSP)
	ADD R8, R5, R9
	MOVD R9, 16(RSP)
	MOVD R7, 24(RSP)
	CALL ·SiGLUTensorsF32NEON(SB)
	LSL $3, R7, R10
	ADD R10, R5
	LSL $2, R7, R10
	ADD R10, R4
	SUB $1, R6
	CBNZ R6, siglu_packed_neon_row
siglu_packed_neon_done:
	RET

// func SeGLUPackedF32NEON(dst, packed *float32, batch, halfCount int)
TEXT ·SeGLUPackedF32NEON(SB), NOSPLIT, $32-32
	MOVD dst+0(FP), R4
	MOVD packed+8(FP), R5
	MOVD batch+16(FP), R6
	MOVD halfCount+24(FP), R7
	CBZ R6, seglu_packed_neon_done
	LSL $2, R7, R8
seglu_packed_neon_row:
	MOVD R4, 0(RSP)
	MOVD R5, 8(RSP)
	ADD R8, R5, R9
	MOVD R9, 16(RSP)
	MOVD R7, 24(RSP)
	CALL ·SeGLUTensorsF32NEON(SB)
	LSL $3, R7, R10
	ADD R10, R5
	LSL $2, R7, R10
	ADD R10, R4
	SUB $1, R6
	CBNZ R6, seglu_packed_neon_row
seglu_packed_neon_done:
	RET

// func GeGLUPackedF32NEON(dst, packed *float32, batch, halfCount int)
TEXT ·GeGLUPackedF32NEON(SB), NOSPLIT, $32-32
	MOVD dst+0(FP), R4
	MOVD packed+8(FP), R5
	MOVD batch+16(FP), R6
	MOVD halfCount+24(FP), R7
	CBZ R6, geglu_packed_neon_done
	LSL $2, R7, R8
geglu_packed_neon_row:
	MOVD R4, 0(RSP)
	MOVD R5, 8(RSP)
	ADD R8, R5, R9
	MOVD R9, 16(RSP)
	MOVD R7, 24(RSP)
	CALL ·GeGLUTensorsF32NEON(SB)
	LSL $3, R7, R10
	ADD R10, R5
	LSL $2, R7, R10
	ADD R10, R4
	SUB $1, R6
	CBNZ R6, geglu_packed_neon_row
geglu_packed_neon_done:
	RET

// func GeGLUTanhPackedF32NEON(dst, packed *float32, batch, halfCount int)
TEXT ·GeGLUTanhPackedF32NEON(SB), NOSPLIT, $32-32
	MOVD dst+0(FP), R4
	MOVD packed+8(FP), R5
	MOVD batch+16(FP), R6
	MOVD halfCount+24(FP), R7
	CBZ R6, geglu_tanh_packed_neon_done
	LSL $2, R7, R8
geglu_tanh_packed_neon_row:
	MOVD R4, 0(RSP)
	MOVD R5, 8(RSP)
	ADD R8, R5, R9
	MOVD R9, 16(RSP)
	MOVD R7, 24(RSP)
	CALL ·GeGLUTanhTensorsF32NEON(SB)
	LSL $3, R7, R10
	ADD R10, R5
	LSL $2, R7, R10
	ADD R10, R4
	SUB $1, R6
	CBNZ R6, geglu_tanh_packed_neon_row
geglu_tanh_packed_neon_done:
	RET
