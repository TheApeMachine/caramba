#include "textflag.h"

// func PCPredictionErrorFloat32AVX512Asm(observed, predicted, output *float32, count int)
TEXT ·PCPredictionErrorFloat32AVX512Asm(SB), NOSPLIT, $0-28
	MOVQ observed+0(FP), SI
	MOVQ predicted+8(FP), DI
	MOVQ output+16(FP), BX
	MOVQ count+24(FP), CX

pc_pe_w16:
	CMPQ CX, $16
	JL   pc_pe_w8

	VMOVUPS (SI), Z0
	VMOVUPS (DI), Z1
	VSUBPS  Z1, Z0, Z0
	VMOVUPS Z0, (BX)

	ADDQ $64, SI
	ADDQ $64, DI
	ADDQ $64, BX
	SUBQ $16, CX
	JMP  pc_pe_w16

pc_pe_w8:
	CMPQ CX, $8
	JL   pc_pe_w4

	VMOVUPS (SI), Y0
	VMOVUPS (DI), Y1
	VSUBPS  Y1, Y0, Y0
	VMOVUPS Y0, (BX)

	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $32, BX
	SUBQ $8, CX
	JMP  pc_pe_w8

pc_pe_w4:
	CMPQ CX, $4
	JL   pc_pe_w4_tail

	VMOVUPS (SI), X0
	VMOVUPS (DI), X1
	VSUBPS  X1, X0, X0
	VMOVUPS X0, (BX)

	ADDQ $16, SI
	ADDQ $16, DI
	ADDQ $16, BX
	SUBQ $4, CX
	JMP  pc_pe_w4

pc_pe_w4_tail:
	TESTQ CX, CX
	JZ   pc_pe_done

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y0
	VMOVDQU32 (DI), K7, Y1
	VSUBPS  Y1, Y0, K7, Y0
	VMOVDQU32 Y0, K7, (BX)

pc_pe_done:
	RET

// func PCPredictionFloat32AVX512Asm(weights, representation, output *float32, outDim, inDim int)
TEXT ·PCPredictionFloat32AVX512Asm(SB), NOSPLIT, $0-36
	MOVQ weights+0(FP), R11
	MOVQ representation+8(FP), R12
	MOVQ output+16(FP), DI
	MOVQ outDim+24(FP), R9
	MOVQ inDim+32(FP), R10

	MOVQ R10, R13
	SHLQ $2, R13

pc_pred_row:
	TESTQ R9, R9
	JZ   pc_pred_done

	MOVQ R11, SI
	MOVQ R12, DX
	MOVQ R10, CX

	VXORPD Y0, Y0, Y0

pc_pred_dot_w8:
	CMPQ CX, $8
	JL   pc_pred_dot_w4

	VMOVUPS Y1, (SI)
	VMOVUPS Y2, (DX)
	VEXTRACTF128 $0, Y1, X3
	VEXTRACTF128 $0, Y2, X4
	VCVTPS2PD X3, Y5
	VCVTPS2PD X4, Y6
	VFMADD231PD Y0, Y6, Y5
	VEXTRACTF128 $1, Y1, X3
	VEXTRACTF128 $1, Y2, X4
	VCVTPS2PD X3, Y5
	VCVTPS2PD X4, Y6
	VFMADD231PD Y0, Y6, Y5

	ADDQ $32, SI
	ADDQ $32, DX
	SUBQ $8, CX
	JMP  pc_pred_dot_w8

pc_pred_dot_w4:
	CMPQ CX, $4
	JL   pc_pred_dot_w4_tail

	VMOVUPS X1, (SI)
	VMOVUPS X2, (DX)
	VCVTPS2PD X1, Y5
	VCVTPS2PD X2, Y6
	VFMADD231PD Y0, Y6, Y5

	ADDQ $16, SI
	ADDQ $16, DX
	SUBQ $4, CX
	JMP  pc_pred_dot_w4

pc_pred_dot_w4_tail:
	TESTQ CX, CX
	JZ   pc_pred_dot_reduce

	MOVQ  CX, AX
	MOVQ  $1, BX
	SHLQ  CL, BX
	DECQ  BX
	KMOVQ BX, K7

	VMOVDQU32 (SI), K7, Y1
	VMOVDQU32 (DX), K7, Y2
	VEXTRACTF128 $0, Y1, X3
	VEXTRACTF128 $0, Y2, X4
	VCVTPS2PD X3, Y5
	VCVTPS2PD X4, Y6
	VFMADD231PD Y6, Y5, K7, Y0

pc_pred_dot_reduce:
	VHADDPD Y1, Y0, Y0
	VHADDPD Y1, Y1, Y1
	VEXTRACTF128 $0, Y1, X0
	CVTSD2SS X0, X0
	MOVSS X0, (DI)

	ADDQ $4, DI
	ADDQ R13, R11
	DECQ R9
	JMP  pc_pred_row

pc_pred_done:
	RET

// func PCUpdateRepresentationFloat32AVX512Asm(weights, representation, predictionError, output *float32, learningRate float32, outDim, inDim int)
TEXT ·PCUpdateRepresentationFloat32AVX512Asm(SB), NOSPLIT, $0-48
	MOVQ weights+0(FP), R11
	MOVQ representation+8(FP), R12
	MOVQ predictionError+16(FP), R13
	MOVQ output+24(FP), DI
	MOVSS learningRate+32(FP), X15
	MOVQ outDim+36(FP), R9
	MOVQ inDim+40(FP), R14

	MOVQ R12, SI
	MOVQ DI, BX
	MOVQ R14, CX

pc_ur_copy_w16:
	CMPQ CX, $16
	JL   pc_ur_copy_w8

	VMOVUPS (SI), Y0
	VMOVUPS Y0, (BX)
	VMOVUPS 32(SI), Y1
	VMOVUPS Y1, 32(BX)

	ADDQ $64, SI
	ADDQ $64, BX
	SUBQ $16, CX
	JMP  pc_ur_copy_w16

pc_ur_copy_w8:
	CMPQ CX, $8
	JL   pc_ur_copy_w4

	VMOVUPS (SI), Y0
	VMOVUPS Y0, (BX)

	ADDQ $32, SI
	ADDQ $32, BX
	SUBQ $8, CX
	JMP  pc_ur_copy_w8

pc_ur_copy_w4:
	CMPQ CX, $4
	JL   pc_ur_copy_w4_tail

	VMOVUPS (SI), X0
	VMOVUPS X0, (BX)

	ADDQ $16, SI
	ADDQ $16, BX
	SUBQ $4, CX
	JMP  pc_ur_copy_w4

pc_ur_copy_w4_tail:
	TESTQ CX, CX
	JZ   pc_ur_rows

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y0
	VMOVDQU32 Y0, K7, (BX)

pc_ur_rows:
	MOVQ R14, R15
	SHLQ $2, R15

pc_ur_row:
	TESTQ R9, R9
	JZ   pc_ur_done

	MOVSS (R13), X0
	VMULSS X15, X0, X0
	VBROADCASTSS X0, Z0

	MOVQ R11, SI
	MOVQ DI, BX
	MOVQ R14, CX

pc_ur_w16:
	CMPQ CX, $16
	JL   pc_ur_w8

	VMOVUPS (SI), Z1
	VMOVUPS (BX), Z2
	VMULPS  Z0, Z1, Z1
	VADDPS  Z1, Z2, Z2
	VMOVUPS Z2, (BX)

	ADDQ $64, SI
	ADDQ $64, BX
	SUBQ $16, CX
	JMP  pc_ur_w16

pc_ur_w8:
	CMPQ CX, $8
	JL   pc_ur_w4

	VMOVUPS (SI), Y1
	VMOVUPS (BX), Y2
	VMULPS  Y0, Y1, Y1
	VADDPS  Y1, Y2, Y2
	VMOVUPS Y2, (BX)

	ADDQ $32, SI
	ADDQ $32, BX
	SUBQ $8, CX
	JMP  pc_ur_w8

pc_ur_w4:
	CMPQ CX, $4
	JL   pc_ur_w4_tail

	VMOVUPS (SI), X1
	VMOVUPS (BX), X2
	VMULPS  X0, X1, X1
	VADDPS  X1, X2, X2
	VMOVUPS X2, (BX)

	ADDQ $16, SI
	ADDQ $16, BX
	SUBQ $4, CX
	JMP  pc_ur_w4

pc_ur_w4_tail:
	TESTQ CX, CX
	JZ   pc_ur_next_row

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y1
	VMOVDQU32 (BX), K7, Y2
	VMULPS  Y0, Y1, K7, Y1
	VADDPS  Y1, Y2, K7, Y2
	VMOVDQU32 Y2, K7, (BX)

pc_ur_next_row:
	ADDQ $4, R13
	ADDQ R15, R11
	DECQ R9
	JMP  pc_ur_row

pc_ur_done:
	RET

// func PCUpdateWeightsFloat32AVX512Asm(weights, representation, predictionError, output *float32, learningRate float32, outDim, inDim int)
TEXT ·PCUpdateWeightsFloat32AVX512Asm(SB), NOSPLIT, $0-48
	MOVQ weights+0(FP), R11
	MOVQ representation+8(FP), R12
	MOVQ predictionError+16(FP), R13
	MOVQ output+24(FP), DI
	MOVSS learningRate+32(FP), X15
	MOVQ outDim+36(FP), R9
	MOVQ inDim+40(FP), R14

	MOVQ R11, SI
	MOVQ DI, BX
	MOVQ R9, CX
	IMULQ R14, CX

pc_uw_copy_w16:
	CMPQ CX, $16
	JL   pc_uw_copy_w8

	VMOVUPS (SI), Y0
	VMOVUPS Y0, (BX)
	VMOVUPS 32(SI), Y1
	VMOVUPS Y1, 32(BX)

	ADDQ $64, SI
	ADDQ $64, BX
	SUBQ $16, CX
	JMP  pc_uw_copy_w16

pc_uw_copy_w8:
	CMPQ CX, $8
	JL   pc_uw_copy_w4

	VMOVUPS (SI), Y0
	VMOVUPS Y0, (BX)

	ADDQ $32, SI
	ADDQ $32, BX
	SUBQ $8, CX
	JMP  pc_uw_copy_w8

pc_uw_copy_w4:
	CMPQ CX, $4
	JL   pc_uw_copy_w4_tail

	VMOVUPS (SI), X0
	VMOVUPS X0, (BX)

	ADDQ $16, SI
	ADDQ $16, BX
	SUBQ $4, CX
	JMP  pc_uw_copy_w4

pc_uw_copy_w4_tail:
	TESTQ CX, CX
	JZ   pc_uw_rows

	MOVQ  CX, DX
	MOVQ  $1, AX
	SHLQ  CL, AX
	DECQ  AX
	KMOVQ AX, K7

	VMOVDQU32 (SI), K7, Y0
	VMOVDQU32 Y0, K7, (BX)

pc_uw_rows:
	MOVQ R14, R15
	SHLQ $2, R15

pc_uw_row:
	TESTQ R9, R9
	JZ   pc_uw_done

	MOVSS (R13), X0
	VMULSS X15, X0, X0
	VBROADCASTSS X0, Z0

	MOVQ R11, SI
	MOVQ R12, DX
	MOVQ DI, BX
	MOVQ R14, CX

pc_uw_w16:
	CMPQ CX, $16
	JL   pc_uw_w8

	VMOVUPS (DX), Z1
	VMOVUPS (BX), Z2
	VMULPS  Z0, Z1, Z1
	VADDPS  Z1, Z2, Z2
	VMOVUPS Z2, (BX)

	ADDQ $64, DX
	ADDQ $64, BX
	SUBQ $16, CX
	JMP  pc_uw_w16

pc_uw_w8:
	CMPQ CX, $8
	JL   pc_uw_w4

	VMOVUPS (DX), Y1
	VMOVUPS (BX), Y2
	VMULPS  Y0, Y1, Y1
	VADDPS  Y1, Y2, Y2
	VMOVUPS Y2, (BX)

	ADDQ $32, DX
	ADDQ $32, BX
	SUBQ $8, CX
	JMP  pc_uw_w8

pc_uw_w4:
	CMPQ CX, $4
	JL   pc_uw_w4_tail

	VMOVUPS (DX), X1
	VMOVUPS (BX), X2
	VMULPS  X0, X1, X1
	VADDPS  X1, X2, X2
	VMOVUPS X2, (BX)

	ADDQ $16, DX
	ADDQ $16, BX
	SUBQ $4, CX
	JMP  pc_uw_w4

pc_uw_w4_tail:
	TESTQ CX, CX
	JZ   pc_uw_next_row

	MOVQ  CX, AX
	MOVQ  $1, R8
	SHLQ  CL, R8
	DECQ  R8
	KMOVQ R8, K7

	VMOVDQU32 (DX), K7, Y1
	VMOVDQU32 (BX), K7, Y2
	VMULPS  Y0, Y1, K7, Y1
	VADDPS  Y1, Y2, K7, Y2
	VMOVDQU32 Y2, K7, (BX)

pc_uw_next_row:
	ADDQ $4, R13
	ADDQ R15, R11
	ADDQ R15, DI
	DECQ R9
	JMP  pc_uw_row

pc_uw_done:
	RET
