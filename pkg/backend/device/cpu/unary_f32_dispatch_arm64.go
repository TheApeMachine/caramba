//go:build arm64

package cpu

func AbsFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	AbsFloat32NEONAsm(&dst[0], &src[0], len(dst))
}

func NegFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	NegFloat32NEONAsm(&dst[0], &src[0], len(dst))
}

func SqrtFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	SqrtFloat32NEONAsm(&dst[0], &src[0], len(dst))
}

func ReluFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	ReluFloat32NEONAsm(&dst[0], &src[0], len(dst))
}
