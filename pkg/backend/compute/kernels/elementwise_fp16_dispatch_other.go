//go:build !amd64 && !arm64

package kernels

import "github.com/theapemachine/caramba/pkg/dtype"

func addFloat16Native(dst, left, right []dtype.F16) {
	for index := range dst {
		dst[index] = dtype.Fromfloat32(left[index].Float32() + right[index].Float32())
	}
}

func subFloat16Native(dst, left, right []dtype.F16) {
	for index := range dst {
		dst[index] = dtype.Fromfloat32(left[index].Float32() - right[index].Float32())
	}
}

func mulFloat16Native(dst, left, right []dtype.F16) {
	for index := range dst {
		dst[index] = dtype.Fromfloat32(left[index].Float32() * right[index].Float32())
	}
}

func divFloat16Native(dst, left, right []dtype.F16) {
	for index := range dst {
		dst[index] = dtype.Fromfloat32(left[index].Float32() / right[index].Float32())
	}
}

func maxFloat16Native(dst, left, right []dtype.F16) {
	for index := range dst {
		leftValue := left[index].Float32()
		rightValue := right[index].Float32()
		dst[index] = dtype.Fromfloat32(rightValue)

		if leftValue > rightValue {
			dst[index] = dtype.Fromfloat32(leftValue)
		}
	}
}

func minFloat16Native(dst, left, right []dtype.F16) {
	for index := range dst {
		leftValue := left[index].Float32()
		rightValue := right[index].Float32()
		dst[index] = dtype.Fromfloat32(rightValue)

		if leftValue < rightValue {
			dst[index] = dtype.Fromfloat32(leftValue)
		}
	}
}
