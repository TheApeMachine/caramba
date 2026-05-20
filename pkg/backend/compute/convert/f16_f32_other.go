//go:build !amd64 && !arm64

package convert

import "github.com/theapemachine/manifesto/dtype"

/*
Fallback dispatcher for F16↔F32 on platforms outside amd64 / arm64.
Routes straight to the scalar reference.
*/

func float16ToFloat32Native(dst []float32, src []dtype.F16) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	return float16ToFloat32Scalar(dst, src)
}

func float32ToFloat16Native(dst []dtype.F16, src []float32) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	return float32ToFloat16Scalar(dst, src)
}
