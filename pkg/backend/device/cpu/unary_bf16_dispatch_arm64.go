//go:build arm64

package cpu

import "github.com/theapemachine/caramba/pkg/dtype"

func AbsBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	AbsBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

func NegBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	NegBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

func SqrtBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	SqrtBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

func ReluBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	ReluBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}
