//go:build !amd64 && !arm64

package convert

import "github.com/theapemachine/caramba/pkg/dtype"

/*
Fallback dispatcher for platforms outside amd64 / arm64. Routes
straight to the scalar reference; no SIMD available.
*/

func bfloat16ToFloat32(dst []float32, src []dtype.BF16) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	return bfloat16ToFloat32Scalar(dst, src)
}

func float32ToBFloat16(dst []dtype.BF16, src []float32) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	return float32ToBFloat16Scalar(dst, src)
}
