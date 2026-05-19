//go:build arm64

package cpu

import "github.com/theapemachine/caramba/pkg/dtype"

func AddFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	AddFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func SubFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	SubFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func MulFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	MulFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func DivFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	DivFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func MaxFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	MaxFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func MinFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	MinFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}
