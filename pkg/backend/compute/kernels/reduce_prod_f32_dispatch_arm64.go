//go:build arm64

package kernels

func reduceProdFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	return reduceProdFloat32NEONAsm(&values[0], len(values))
}
