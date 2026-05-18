//go:build arm64

package kernels

func axpyFloat32Native(dst []float32, src []float32, alpha float32) {
	if len(dst) == 0 {
		return
	}

	axpyFloat32NEONAsm(&dst[0], &src[0], alpha, len(dst))
}
