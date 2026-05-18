//go:build !amd64 && !arm64

package kernels

import "github.com/theapemachine/caramba/pkg/dtype"

func addBFloat16Native(dst, left, right []dtype.BF16) {
	for index := range dst {
		sum := (&left[index]).Float32() + (&right[index]).Float32()
		dst[index] = dtype.NewBfloat16FromFloat32(sum)
	}
}

func subBFloat16Native(dst, left, right []dtype.BF16) {
	for index := range dst {
		diff := (&left[index]).Float32() - (&right[index]).Float32()
		dst[index] = dtype.NewBfloat16FromFloat32(diff)
	}
}

func mulBFloat16Native(dst, left, right []dtype.BF16) {
	for index := range dst {
		product := (&left[index]).Float32() * (&right[index]).Float32()
		dst[index] = dtype.NewBfloat16FromFloat32(product)
	}
}

func divBFloat16Native(dst, left, right []dtype.BF16) {
	for index := range dst {
		quotient := (&left[index]).Float32() / (&right[index]).Float32()
		dst[index] = dtype.NewBfloat16FromFloat32(quotient)
	}
}

func maxBFloat16Native(dst, left, right []dtype.BF16) {
	for index := range dst {
		leftValue := (&left[index]).Float32()
		rightValue := (&right[index]).Float32()
		dst[index] = dtype.NewBfloat16FromFloat32(rightValue)

		if leftValue > rightValue {
			dst[index] = dtype.NewBfloat16FromFloat32(leftValue)
		}
	}
}

func minBFloat16Native(dst, left, right []dtype.BF16) {
	for index := range dst {
		leftValue := (&left[index]).Float32()
		rightValue := (&right[index]).Float32()
		dst[index] = dtype.NewBfloat16FromFloat32(rightValue)

		if leftValue < rightValue {
			dst[index] = dtype.NewBfloat16FromFloat32(leftValue)
		}
	}
}
