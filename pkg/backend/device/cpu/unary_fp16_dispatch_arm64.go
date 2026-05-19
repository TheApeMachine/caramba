//go:build arm64

package cpu

import "github.com/theapemachine/caramba/pkg/dtype"

func AbsFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	AbsFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

func NegFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	NegFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

func SqrtFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	SqrtFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

func ReluFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	ReluFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}
