package elementwise

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func dispatchScalarUnary(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	apply func(float32) float32,
) {
	dispatchUnary(
		dst, src, count, format,
		func(dst, src unsafe.Pointer, count int) {
			runUnaryScalarF32(dst, src, count, apply)
		},
		apply,
	)
}

func Square(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return value * value
	})
}

func Rsqrt(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(1.0 / math.Sqrt(float64(value)))
	})
}

func Recip(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return 1 / value
	})
}

func Exp(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(math.Exp(float64(value)))
	})
}

func Log(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(math.Log(float64(value)))
	})
}

func Log1p(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(math.Log1p(float64(value)))
	})
}

func Expm1(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(math.Expm1(float64(value)))
	})
}

func Sin(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(math.Sin(float64(value)))
	})
}

func Cos(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(math.Cos(float64(value)))
	})
}

func Tan(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(math.Tan(float64(value)))
	})
}

func Asin(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(math.Asin(float64(value)))
	})
}

func Acos(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(math.Acos(float64(value)))
	})
}

func Atan(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(math.Atan(float64(value)))
	})
}

func Sinh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(math.Sinh(float64(value)))
	})
}

func Cosh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(math.Cosh(float64(value)))
	})
}

func Tanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(math.Tanh(float64(value)))
	})
}

func Erf(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(math.Erf(float64(value)))
	})
}

func Erfc(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(math.Erfc(float64(value)))
	})
}

func Ceil(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(math.Ceil(float64(value)))
	})
}

func Floor(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(math.Floor(float64(value)))
	})
}

func Round(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(math.Round(float64(value)))
	})
}

func Trunc(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		return float32(math.Trunc(float64(value)))
	})
}

func Sign(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchScalarUnary(dst, src, count, format, func(value float32) float32 {
		switch {
		case value > 0:
			return 1
		case value < 0:
			return -1
		}

		return 0
	})
}
