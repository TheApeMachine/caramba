//go:build arm64

package kernels

func sigmoidFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	sigmoidFloat32NEONAsm(&dst[0], &src[0], len(dst))
}

func siluFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	siluFloat32NEONAsm(&dst[0], &src[0], len(dst))
}

func tanhFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	tanhFloat32NEONAsm(&dst[0], &src[0], len(dst))
}
