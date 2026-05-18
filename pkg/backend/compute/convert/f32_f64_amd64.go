//go:build amd64

package convert

/*
amd64 dispatcher for F32↔F64. AVX-512 / AVX2 / SSE2 VCVTPS2PD /
VCVTPD2PS would land in .s files in a hardware-verified session;
today this routes through the scalar reference.
*/

func float32ToFloat64Native(dst []float64, src []float32) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	for index, value := range src {
		dst[index] = float64(value)
	}

	return nil
}

func float64ToFloat32Native(dst []float32, src []float64) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	for index, value := range src {
		dst[index] = float32(value)
	}

	return nil
}
