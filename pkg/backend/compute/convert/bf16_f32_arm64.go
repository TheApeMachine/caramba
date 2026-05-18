//go:build arm64

package convert

import "github.com/theapemachine/caramba/pkg/dtype"

/*
ARM64 dispatchers for BF16↔F32. Routes through a NEON loop with
scalar tail. The real assembly (with bfcvt/bfcvtn on armv8.6+, or
manual shift on older cores) lands in bf16_f32_arm64.s in a
hardware-verified session; this Go-side dispatcher pins the call
shape so .s files drop in without changing the public surface.
*/

func bfloat16ToFloat32(dst []float32, src []dtype.BF16) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	tail := bfloat16ToFloat32NEONLoop(dst, src)

	if tail < len(src) {
		return bfloat16ToFloat32Scalar(dst[tail:], src[tail:])
	}

	return nil
}

func float32ToBFloat16(dst []dtype.BF16, src []float32) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	tail := float32ToBFloat16NEONLoop(dst, src)

	if tail < len(src) {
		return float32ToBFloat16Scalar(dst[tail:], src[tail:])
	}

	return nil
}

// bfloat16ToFloat32NEONLoop is the NEON loop entry point. Returns
// the number of elements consumed; the caller handles the tail with
// the scalar reference. The Go-side stub here forwards to the scalar
// implementation so the package builds before bf16_f32_arm64.s lands.
func bfloat16ToFloat32NEONLoop(dst []float32, src []dtype.BF16) int {
	_ = bfloat16ToFloat32Scalar(dst, src)
	return len(src)
}

func float32ToBFloat16NEONLoop(dst []dtype.BF16, src []float32) int {
	_ = float32ToBFloat16Scalar(dst, src)
	return len(src)
}
