//go:build arm64

package kernels

import "github.com/theapemachine/caramba/pkg/dtype"

func addFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	addFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func subFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	subFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func mulFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	mulFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func divFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	divFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func maxFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	maxFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func minFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	minFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}
