//go:build amd64

package cpu

import "github.com/theapemachine/caramba/pkg/dtype"

/*
amd64 dispatcher for fp16 elementwise binary ops. F16C (Ivy Bridge+)
provides VCVTPH2PS / VCVTPS2PH for widening / narrowing, and AVX2 /
AVX-512 do the f32 arithmetic. Hardware-verified .s paths land in a
follow-on session; today this routes through the scalar reference.
*/

func AddFloat16Native(dst, left, right []dtype.F16) {
	for index := range dst {
		dst[index] = dtype.Fromfloat32(left[index].Float32() + right[index].Float32())
	}
}

func SubFloat16Native(dst, left, right []dtype.F16) {
	for index := range dst {
		dst[index] = dtype.Fromfloat32(left[index].Float32() - right[index].Float32())
	}
}

func MulFloat16Native(dst, left, right []dtype.F16) {
	for index := range dst {
		dst[index] = dtype.Fromfloat32(left[index].Float32() * right[index].Float32())
	}
}

func DivFloat16Native(dst, left, right []dtype.F16) {
	for index := range dst {
		dst[index] = dtype.Fromfloat32(left[index].Float32() / right[index].Float32())
	}
}

func MaxFloat16Native(dst, left, right []dtype.F16) {
	for index := range dst {
		leftValue := left[index].Float32()
		rightValue := right[index].Float32()
		dst[index] = dtype.Fromfloat32(rightValue)

		if leftValue > rightValue {
			dst[index] = dtype.Fromfloat32(leftValue)
		}
	}
}

func MinFloat16Native(dst, left, right []dtype.F16) {
	for index := range dst {
		leftValue := left[index].Float32()
		rightValue := right[index].Float32()
		dst[index] = dtype.Fromfloat32(rightValue)

		if leftValue < rightValue {
			dst[index] = dtype.Fromfloat32(leftValue)
		}
	}
}
