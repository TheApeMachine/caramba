#include "textflag.h"

// FADD/FSUB/FMUL of two D2 vectors. Macro arg order (m, n, d) encodes
// Vd = Vn op Vm, mirroring the convention used in
// pkg/backend/compute/cpu/operation/math/primitives_neon_arm64.s so the
// kernels below read the same way as the rest of the codebase.
#define VFADD_D2(m, n, d) WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_D2(m, n, d) WORD $(0x4EE0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_D2(m, n, d) WORD $(0x6E60DC00 | ((m) << 16) | ((n) << 5) | (d))

// laplacianAxisSetNEON(out, left, center, right []float64, invH2 float64)
// out[i] = (left[i] + right[i] - 2*center[i]) * invH2
// ABI0:
//   out+0(FP),    out_len+8(FP),    out_cap+16(FP)
//   left+24(FP),  left_len+32(FP),  left_cap+40(FP)
//   center+48(FP),center_len+56(FP),center_cap+64(FP)
//   right+72(FP), right_len+80(FP), right_cap+88(FP)
//   invH2+96(FP)
TEXT ·laplacianAxisSetNEON(SB), NOSPLIT, $8-104
	MOVD  out+0(FP), R0
	MOVD  out_len+8(FP), R3
	MOVD  left+24(FP), R1
	MOVD  center+48(FP), R2
	MOVD  right+72(FP), R4
	FMOVD invH2+96(FP), F15
	FMOVD F15, 0(RSP)
	VLD1R (RSP), [V15.D2]
	LSR   $1, R3, R5
	CBZ   R5, done_set_neon
loop_set_neon:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R4), [V1.D2]
	VFADD_D2(1, 0, 0)
	VLD1.P 16(R2), [V2.D2]
	VFADD_D2(2, 2, 2)
	VFSUB_D2(2, 0, 0)
	VFMUL_D2(15, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS   $1, R5, R5
	BNE    loop_set_neon
done_set_neon:
	RET

// laplacianAxisAccNEON(out, left, center, right []float64, invH2 float64)
// out[i] += (left[i] + right[i] - 2*center[i]) * invH2
TEXT ·laplacianAxisAccNEON(SB), NOSPLIT, $8-104
	MOVD  out+0(FP), R0
	MOVD  out_len+8(FP), R3
	MOVD  left+24(FP), R1
	MOVD  center+48(FP), R2
	MOVD  right+72(FP), R4
	FMOVD invH2+96(FP), F15
	FMOVD F15, 0(RSP)
	VLD1R (RSP), [V15.D2]
	LSR   $1, R3, R5
	CBZ   R5, done_acc_neon
loop_acc_neon:
	VLD1.P 16(R1), [V0.D2]
	VLD1.P 16(R4), [V1.D2]
	VFADD_D2(1, 0, 0)
	VLD1.P 16(R2), [V2.D2]
	VFADD_D2(2, 2, 2)
	VFSUB_D2(2, 0, 0)
	VFMUL_D2(15, 0, 0)
	VLD1  (R0), [V3.D2]
	VFADD_D2(3, 0, 0)
	VST1.P [V0.D2], 16(R0)
	SUBS   $1, R5, R5
	BNE    loop_acc_neon
done_acc_neon:
	RET
