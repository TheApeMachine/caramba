//go:build arm64

package cpu

func ExpFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	ExpFloat32NEONAsm(&dst[0], &src[0], len(dst))
}
