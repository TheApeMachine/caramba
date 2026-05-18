//go:build amd64

package kernels

import "github.com/theapemachine/caramba/pkg/dtype"

/*
amd64 dispatcher for bf16 elementwise binary ops. AVX-512-BF16 has
native VDPBF16PS for accumulate and VCVTNE2PS2BF16 for narrowing;
AVX2 + F16C handles widening with VPMOVZXWD + VPSLLD by 16 and
narrowing with VPSRAD by 16 + VPACKSSDW. Hardware-verified paths
land in .s files in a follow-on session; today this routes through
the scalar reference, which is itself bit-exact (bf16 widen/narrow
is pure bit-shuffle).
*/

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
