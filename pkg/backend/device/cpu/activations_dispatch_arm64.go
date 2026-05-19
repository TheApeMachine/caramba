//go:build arm64

package cpu

import "github.com/theapemachine/caramba/pkg/backend/device/cpu/neon"

func SigmoidFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	neon.SigmoidFloat32NEONAsm(&dst[0], &src[0], len(dst))
}

func SiluFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	neon.SiluFloat32NEONAsm(&dst[0], &src[0], len(dst))
}

func TanhFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	neon.TanhFloat32NEONAsm(&dst[0], &src[0], len(dst))
}
