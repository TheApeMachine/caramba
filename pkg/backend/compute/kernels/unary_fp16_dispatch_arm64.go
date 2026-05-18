//go:build arm64

package kernels

import "github.com/theapemachine/caramba/pkg/dtype"

func absFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	absFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

func negFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	negFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

func sqrtFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	sqrtFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

func reluFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	reluFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}
