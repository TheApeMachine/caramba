//go:build arm64

package kernels

import "github.com/theapemachine/caramba/pkg/dtype"

func absBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	absBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

func negBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	negBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

func sqrtBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	sqrtBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

func reluBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	reluBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}
