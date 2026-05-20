package elementwise

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func Add(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	if format == dtype.Float64 {
		if count == 0 {
			return
		}

		runAddF64(dst, left, right, count)

		return
	}

	dispatchBinary(
		dst, left, right, count, format, runAddF32,
		func(leftValue, rightValue float32) float32 { return leftValue + rightValue },
	)
}

func Sub(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	dispatchBinary(
		dst, left, right, count, format, runSubF32,
		func(leftValue, rightValue float32) float32 { return leftValue - rightValue },
	)
}

func Mul(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	dispatchBinary(
		dst, left, right, count, format, runMulF32,
		func(leftValue, rightValue float32) float32 { return leftValue * rightValue },
	)
}

func Div(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	dispatchBinary(
		dst, left, right, count, format, runDivF32,
		func(leftValue, rightValue float32) float32 { return leftValue / rightValue },
	)
}

func Max(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	dispatchBinary(
		dst, left, right, count, format, runMaxF32,
		func(leftValue, rightValue float32) float32 {
			if leftValue > rightValue {
				return leftValue
			}

			return rightValue
		},
	)
}

func Min(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	dispatchBinary(
		dst, left, right, count, format, runMinF32,
		func(leftValue, rightValue float32) float32 {
			if leftValue < rightValue {
				return leftValue
			}

			return rightValue
		},
	)
}

func Abs(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchUnary(
		dst, src, count, format, runAbsF32,
		func(value float32) float32 { return float32(math.Abs(float64(value))) },
	)
}

func Neg(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchUnary(
		dst, src, count, format, runNegF32,
		func(value float32) float32 { return -value },
	)
}

func Sqrt(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchUnary(
		dst, src, count, format, runSqrtF32,
		func(value float32) float32 { return float32(math.Sqrt(float64(value))) },
	)
}

func ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchUnary(
		dst, src, count, format, runReluF32,
		func(value float32) float32 {
			if value > 0 {
				return value
			}

			return 0
		},
	)
}

func Axpy(y, x unsafe.Pointer, count int, alpha float32, format dtype.DType) {
	dispatchAxpy(y, x, count, alpha, format, runAxpyF32)
}
