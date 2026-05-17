#include "textflag.h"

// laplacianAxisSetAVX2(out, left, center, right []float64, invH2 float64)
// out[i] = (left[i] + right[i] - 2*center[i]) * invH2
// ABI0:
//   out+0(FP),    out_len+8(FP),    out_cap+16(FP)
//   left+24(FP),  left_len+32(FP),  left_cap+40(FP)
//   center+48(FP),center_len+56(FP),center_cap+64(FP)
//   right+72(FP), right_len+80(FP), right_cap+88(FP)
//   invH2+96(FP)
// All slices share the length passed in out_len.
TEXT ·laplacianAxisSetAVX2(SB), NOSPLIT, $0-104
	MOVQ         out+0(FP), DI
	MOVQ         out_len+8(FP), CX
	MOVQ         left+24(FP), R8
	MOVQ         center+48(FP), R9
	MOVQ         right+72(FP), R10
	VBROADCASTSD invH2+96(FP), Y15
	CMPQ         CX, $4
	JL           done_set_avx2
loop_set_avx2:
	VMOVUPD (R8), Y0
	VMOVUPD (R10), Y1
	VADDPD  Y1, Y0, Y0
	VMOVUPD (R9), Y2
	VADDPD  Y2, Y2, Y2
	VSUBPD  Y2, Y0, Y0
	VMULPD  Y15, Y0, Y0
	VMOVUPD Y0, (DI)
	ADDQ    $32, DI
	ADDQ    $32, R8
	ADDQ    $32, R9
	ADDQ    $32, R10
	SUBQ    $4, CX
	CMPQ    CX, $4
	JGE     loop_set_avx2
done_set_avx2:
	VZEROUPPER
	RET

// laplacianAxisAccAVX2(out, left, center, right []float64, invH2 float64)
// out[i] += (left[i] + right[i] - 2*center[i]) * invH2
TEXT ·laplacianAxisAccAVX2(SB), NOSPLIT, $0-104
	MOVQ         out+0(FP), DI
	MOVQ         out_len+8(FP), CX
	MOVQ         left+24(FP), R8
	MOVQ         center+48(FP), R9
	MOVQ         right+72(FP), R10
	VBROADCASTSD invH2+96(FP), Y15
	CMPQ         CX, $4
	JL           done_acc_avx2
loop_acc_avx2:
	VMOVUPD (R8), Y0
	VMOVUPD (R10), Y1
	VADDPD  Y1, Y0, Y0
	VMOVUPD (R9), Y2
	VADDPD  Y2, Y2, Y2
	VSUBPD  Y2, Y0, Y0
	VMULPD  Y15, Y0, Y0
	VMOVUPD (DI), Y3
	VADDPD  Y3, Y0, Y0
	VMOVUPD Y0, (DI)
	ADDQ    $32, DI
	ADDQ    $32, R8
	ADDQ    $32, R9
	ADDQ    $32, R10
	SUBQ    $4, CX
	CMPQ    CX, $4
	JGE     loop_acc_avx2
done_acc_avx2:
	VZEROUPPER
	RET
