//go:build arm64

package kernels

func dotFloat32Native(a, b []float32) float32 {
	if len(a) == 0 {
		return 0
	}

	return dotFloat32NEONAsm(&a[0], &b[0], len(a))
}
