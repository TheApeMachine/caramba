//go:build arm64

package cpu

func AxpyFloat32Native(dst []float32, src []float32, alpha float32) {
	if len(dst) == 0 {
		return
	}

	AxpyFloat32NEONAsm(&dst[0], &src[0], alpha, len(dst))
}
