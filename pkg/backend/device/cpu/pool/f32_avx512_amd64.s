#include "textflag.h"

DATA poolNegInfAVX512<>+0(SB)/4, $0xFF800000
DATA poolOneAVX512<>+0(SB)/4, $0x3F800000
DATA poolQuarterAVX512<>+0(SB)/4, $0x3E800000
GLOBL poolNegInfAVX512<>(SB), RODATA|NOPTR, $4
GLOBL poolOneAVX512<>(SB), RODATA|NOPTR, $4
GLOBL poolQuarterAVX512<>(SB), RODATA|NOPTR, $4

// func MaxPool2DStride1RowAVX512Asm(
//     outRow *float32, input *float32,
//     outCols, kH, kW int,
//     inHStride int,
//     ihStart int,
// )
TEXT ·MaxPool2DStride1RowAVX512Asm(SB), NOSPLIT, $0-56
	MOVQ outRow+0(FP), AX
	MOVQ input+8(FP), BX
	MOVQ outCols+16(FP), CX
	MOVQ kH+24(FP), DX
	MOVQ kW+32(FP), R8
	MOVQ inHStride+40(FP), R9
	MOVQ ihStart+48(FP), R10

	SHLQ $2, R9
	IMULQ R10, R9
	ADDQ R9, BX

	VBROADCASTSS poolNegInfAVX512<>(SB), Y0

mp512_col_loop:
	CMPQ CX, $4
	JL   mp512_done

	VMOVAPS Y0, Y1

	MOVQ DX, R11
	MOVQ BX, R12

mp512_kh_loop:
	TESTQ R11, R11
	JZ   mp512_kh_done

	MOVQ R8, R13
	MOVQ R12, R14

mp512_kw_loop:
	TESTQ R13, R13
	JZ   mp512_kw_done

	VMOVUPS (R14), Y2
	VMAXPS  Y2, Y1, Y1

	ADDQ $4, R14
	DECQ  R13
	JMP   mp512_kw_loop

mp512_kw_done:
	ADDQ R9, R12
	DECQ  R11
	JMP   mp512_kh_loop

mp512_kh_done:
	VMOVUPS Y1, (AX)

	ADDQ $16, AX
	ADDQ $16, BX
	SUBQ $4, CX
	JMP  mp512_col_loop

mp512_done:
	RET

// func AvgPool2DStride1RowAVX512Asm(
//     outRow *float32, input *float32,
//     outCols, kH, kW int,
//     inHStride int,
//     ihStart int,
// )
TEXT ·AvgPool2DStride1RowAVX512Asm(SB), NOSPLIT, $0-56
	MOVQ outRow+0(FP), AX
	MOVQ input+8(FP), BX
	MOVQ outCols+16(FP), CX
	MOVQ kH+24(FP), DX
	MOVQ kW+32(FP), R8
	MOVQ inHStride+40(FP), R9
	MOVQ ihStart+48(FP), R10

	SHLQ $2, R9
	IMULQ R10, R9
	ADDQ R9, BX

	IMULQ R8, DX
	MOVQ  DX, AX
	CVTSQ2SS AX, X14
	VBROADCASTSS poolOneAVX512<>(SB), X13
	VDIVSS  X14, X13, X12
	VBROADCASTSS X12, Y0

	MOVQ kH+24(FP), DX
	MOVQ kW+32(FP), R8

ap512_col_loop:
	CMPQ CX, $4
	JL   ap512_done

	VXORPS Y1, Y1, Y1

	MOVQ DX, R11
	MOVQ BX, R12

ap512_kh_loop:
	TESTQ R11, R11
	JZ   ap512_kh_done

	MOVQ R8, R13
	MOVQ R12, R14

ap512_kw_loop:
	TESTQ R13, R13
	JZ   ap512_kw_done

	VMOVUPS (R14), Y2
	VADDPS  Y2, Y1, Y1

	ADDQ $4, R14
	DECQ  R13
	JMP   ap512_kw_loop

ap512_kw_done:
	ADDQ R9, R12
	DECQ  R11
	JMP   ap512_kh_loop

ap512_kh_done:
	VMULPS Y0, Y1, Y1
	VMOVUPS Y1, (AX)

	ADDQ $16, AX
	ADDQ $16, BX
	SUBQ $4, CX
	JMP  ap512_col_loop

ap512_done:
	RET

// func MaxPool2x2Stride2RowAVX512Asm(
//     outRow *float32, input *float32,
//     outCols, inWidth, ihStart int,
// )
TEXT ·MaxPool2x2Stride2RowAVX512Asm(SB), NOSPLIT, $0-40
	MOVQ outRow+0(FP), AX
	MOVQ input+8(FP), BX
	MOVQ outCols+16(FP), CX
	MOVQ inWidth+24(FP), DX
	MOVQ ihStart+32(FP), R8

	SHLQ $2, DX, R9
	IMULQ R8, R9
	ADDQ R9, BX
	MOVQ BX, R10
	ADDQ R9, R10

mp22512_col_loop:
	CMPQ CX, $4
	JL   mp22512_done

	VMOVUPS (BX), Y0
	VMOVUPS (R10), Y1

	VPERMILPS $0xB1, Y0, Y2
	VMAXPS    Y2, Y0, Y3

	VPERMILPS $0xB1, Y1, Y4
	VMAXPS    Y4, Y1, Y5

	VMAXPS    Y5, Y3, Y6

	VPERMILPS $0x88, Y6, Y7
	VMOVUPS   Y7, (AX)

	ADDQ $32, BX
	ADDQ $32, R10
	ADDQ $16, AX
	SUBQ $4, CX
	JMP  mp22512_col_loop

mp22512_done:
	RET

// func AvgPool2x2Stride2RowAVX512Asm(
//     outRow *float32, input *float32,
//     outCols, inWidth, ihStart int,
// )
TEXT ·AvgPool2x2Stride2RowAVX512Asm(SB), NOSPLIT, $0-40
	MOVQ outRow+0(FP), AX
	MOVQ input+8(FP), BX
	MOVQ outCols+16(FP), CX
	MOVQ inWidth+24(FP), DX
	MOVQ ihStart+32(FP), R8

	SHLQ $2, DX, R9
	IMULQ R8, R9
	ADDQ R9, BX
	MOVQ BX, R10
	ADDQ R9, R10

	VBROADCASTSS poolQuarterAVX512<>(SB), Y15

ap22512_col_loop:
	CMPQ CX, $4
	JL   ap22512_done

	VMOVUPS (BX), Y0
	VMOVUPS (R10), Y1

	VPERMILPS $0xB1, Y0, Y2
	VADDPS    Y2, Y0, Y3

	VPERMILPS $0xB1, Y1, Y4
	VADDPS    Y4, Y1, Y5

	VADDPS    Y5, Y3, Y6

	VPERMILPS $0x88, Y6, Y7
	VMULPS    Y15, Y7, Y7
	VMOVUPS   Y7, (AX)

	ADDQ $32, BX
	ADDQ $32, R10
	ADDQ $16, AX
	SUBQ $4, CX
	JMP  ap22512_col_loop

ap22512_done:
	RET
