//go:build arm64

package kernels

func absFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	absFloat32NEONAsm(&dst[0], &src[0], len(dst))
}

func negFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	negFloat32NEONAsm(&dst[0], &src[0], len(dst))
}

func sqrtFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	sqrtFloat32NEONAsm(&dst[0], &src[0], len(dst))
}

func reluFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	reluFloat32NEONAsm(&dst[0], &src[0], len(dst))
}
