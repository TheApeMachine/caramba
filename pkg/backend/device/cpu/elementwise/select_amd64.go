//go:build amd64

package elementwise

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func AddFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	Add(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float32,
	)
}

func SubFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	Sub(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float32,
	)
}

func MulFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	Mul(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float32,
	)
}

func DivFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	Div(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float32,
	)
}

func MaxFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	Max(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float32,
	)
}

func MinFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	Min(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float32,
	)
}

func AddFloat64Native(dst, left, right []float64) {
	if len(dst) == 0 {
		return
	}

	Add(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float64,
	)
}

func AxpyFloat32Native(y, x []float32, alpha float32) {
	if len(y) == 0 {
		return
	}

	Axpy(
		unsafe.Pointer(&y[0]),
		unsafe.Pointer(&x[0]),
		len(y),
		alpha,
		dtype.Float32,
	)
}

func AbsFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	Abs(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.Float32)
}

func NegFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	Neg(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.Float32)
}

func SqrtFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	Sqrt(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.Float32)
}

func ReluFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	ReLU(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.Float32)
}

func AddBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	Add(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.BFloat16,
	)
}

func SubBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	Sub(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.BFloat16,
	)
}

func MulBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	Mul(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.BFloat16,
	)
}

func DivBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	Div(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.BFloat16,
	)
}

func MaxBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	Max(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.BFloat16,
	)
}

func MinBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	Min(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.BFloat16,
	)
}

func AbsBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	Abs(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.BFloat16)
}

func NegBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	Neg(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.BFloat16)
}

func SqrtBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	Sqrt(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.BFloat16)
}

func ReluBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	ReLU(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.BFloat16)
}

func AddFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	Add(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float16,
	)
}

func SubFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	Sub(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float16,
	)
}

func MulFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	Mul(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float16,
	)
}

func DivFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	Div(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float16,
	)
}

func MaxFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	Max(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float16,
	)
}

func MinFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	Min(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float16,
	)
}

func AbsFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	Abs(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.Float16)
}

func NegFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	Neg(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.Float16)
}

func SqrtFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	Sqrt(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.Float16)
}

func ReluFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	ReLU(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.Float16)
}

var (
	addF32Funcs  = []f32BinaryKernelImpl{{AddF32Generic, "generic", true}}
	subF32Funcs  = []f32BinaryKernelImpl{{SubF32Generic, "generic", true}}
	mulF32Funcs  = []f32BinaryKernelImpl{{MulF32Generic, "generic", true}}
	divF32Funcs  = []f32BinaryKernelImpl{{DivF32Generic, "generic", true}}
	maxF32Funcs  = []f32BinaryKernelImpl{{MaxF32Generic, "generic", true}}
	minF32Funcs  = []f32BinaryKernelImpl{{MinF32Generic, "generic", true}}
	absF32Funcs  = []f32UnaryKernelImpl{{AbsF32Generic, "generic", true}}
	negF32Funcs  = []f32UnaryKernelImpl{{NegF32Generic, "generic", true}}
	sqrtF32Funcs = []f32UnaryKernelImpl{{SqrtF32Generic, "generic", true}}
	reluF32Funcs = []f32UnaryKernelImpl{{ReluF32Generic, "generic", true}}
	axpyF32Funcs = []f32AxpyKernelImpl{{AxpyF32Generic, "generic", true}}
	addF64Funcs  = []f64BinaryKernelImpl{{AddF64Generic, "generic", true}}
)
