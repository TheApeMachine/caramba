package elementwise

import (
	"math"
	"unsafe"
)

func AddF32Generic(dst, left, right *float32, count int) {
	destination := unsafe.Slice(dst, count)
	leftView := unsafe.Slice(left, count)
	rightView := unsafe.Slice(right, count)

	for index := 0; index < count; index++ {
		destination[index] = leftView[index] + rightView[index]
	}
}

func SubF32Generic(dst, left, right *float32, count int) {
	destination := unsafe.Slice(dst, count)
	leftView := unsafe.Slice(left, count)
	rightView := unsafe.Slice(right, count)

	for index := 0; index < count; index++ {
		destination[index] = leftView[index] - rightView[index]
	}
}

func MulF32Generic(dst, left, right *float32, count int) {
	destination := unsafe.Slice(dst, count)
	leftView := unsafe.Slice(left, count)
	rightView := unsafe.Slice(right, count)

	for index := 0; index < count; index++ {
		destination[index] = leftView[index] * rightView[index]
	}
}

func DivF32Generic(dst, left, right *float32, count int) {
	destination := unsafe.Slice(dst, count)
	leftView := unsafe.Slice(left, count)
	rightView := unsafe.Slice(right, count)

	for index := 0; index < count; index++ {
		destination[index] = leftView[index] / rightView[index]
	}
}

func MaxF32Generic(dst, left, right *float32, count int) {
	destination := unsafe.Slice(dst, count)
	leftView := unsafe.Slice(left, count)
	rightView := unsafe.Slice(right, count)

	for index := 0; index < count; index++ {
		destination[index] = rightView[index]

		if leftView[index] > rightView[index] {
			destination[index] = leftView[index]
		}
	}
}

func MinF32Generic(dst, left, right *float32, count int) {
	destination := unsafe.Slice(dst, count)
	leftView := unsafe.Slice(left, count)
	rightView := unsafe.Slice(right, count)

	for index := 0; index < count; index++ {
		destination[index] = rightView[index]

		if leftView[index] < rightView[index] {
			destination[index] = leftView[index]
		}
	}
}

func AbsF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		value := source[index]

		if value < 0 {
			destination[index] = -value
			continue
		}

		destination[index] = value
	}
}

func NegF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = -source[index]
	}
}

func SqrtF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = float32(math.Sqrt(float64(source[index])))
	}
}

func ReluF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		value := source[index]
		destination[index] = 0

		if value > 0 {
			destination[index] = value
		}
	}
}

func AxpyF32Generic(y, x *float32, alpha float32, count int) {
	yView := unsafe.Slice(y, count)
	xView := unsafe.Slice(x, count)

	for index := 0; index < count; index++ {
		yView[index] += alpha * xView[index]
	}
}

func AddF64Generic(dst, left, right *float64, count int) {
	destination := unsafe.Slice(dst, count)
	leftView := unsafe.Slice(left, count)
	rightView := unsafe.Slice(right, count)

	for index := 0; index < count; index++ {
		destination[index] = leftView[index] + rightView[index]
	}
}
