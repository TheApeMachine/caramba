//go:build amd64

package convert

import "github.com/theapemachine/caramba/pkg/dtype"

/*
amd64 dispatcher for F16↔F32. F16C provides VCVTPH2PS / VCVTPS2PH
which would land in .s files in a follow-up session that can verify
on real x86 hardware. Today the dispatcher routes through the scalar
reference; the *Native names are the symbols the public surface in
f16_f32.go calls into.
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
