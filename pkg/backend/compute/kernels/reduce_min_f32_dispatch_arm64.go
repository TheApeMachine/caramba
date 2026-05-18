//go:build arm64

package kernels

func reduceMinFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	return reduceMinFloat32NEONAsm(&values[0], len(values))
}
