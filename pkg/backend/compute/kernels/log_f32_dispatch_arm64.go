//go:build arm64

package kernels

func logFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	logFloat32NEONAsm(&dst[0], &src[0], len(dst))
}
