//go:build arm64

package elementwise

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func AddF32NEON(dst, left, right *float32, count int) {
	if count == 0 {
		return
	}

	AddFloat32NEONAsm(dst, left, right, count)
}

func SubF32NEON(dst, left, right *float32, count int) {
	if count == 0 {
		return
	}

	SubFloat32NEONAsm(dst, left, right, count)
}

func MulF32NEON(dst, left, right *float32, count int) {
	if count == 0 {
		return
	}

	MulFloat32NEONAsm(dst, left, right, count)
}

func DivF32NEON(dst, left, right *float32, count int) {
	if count == 0 {
		return
	}

	DivFloat32NEONAsm(dst, left, right, count)
}

func MaxF32NEON(dst, left, right *float32, count int) {
	if count == 0 {
		return
	}

	MaxFloat32NEONAsm(dst, left, right, count)
}

func MinF32NEON(dst, left, right *float32, count int) {
	if count == 0 {
		return
	}

	MinFloat32NEONAsm(dst, left, right, count)
}

func AbsF32NEON(dst, src *float32, count int) {
	if count == 0 {
		return
	}

	AbsFloat32NEONAsm(dst, src, count)
}

func NegF32NEON(dst, src *float32, count int) {
	if count == 0 {
		return
	}

	NegFloat32NEONAsm(dst, src, count)
}

func SqrtF32NEON(dst, src *float32, count int) {
	if count == 0 {
		return
	}

	SqrtFloat32NEONAsm(dst, src, count)
}

func ReluF32NEON(dst, src *float32, count int) {
	if count == 0 {
		return
	}

	ReluFloat32NEONAsm(dst, src, count)
}

func AxpyF32NEON(y, x *float32, alpha float32, count int) {
	if count == 0 {
		return
	}

	AxpyFloat32NEONAsm(y, x, alpha, count)
}

func AddF64NEON(dst, left, right *float64, count int) {
	if count == 0 {
		return
	}

	AddFloat64NEONAsm(dst, left, right, count)
}

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

	AddBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func SubBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	SubBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func MulBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	MulBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func DivBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	DivBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func MaxBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	MaxBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func MinBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	MinBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func AbsBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	AbsBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

func NegBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	NegBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

func SqrtBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	SqrtBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

func ReluBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	ReluBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

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

func AbsFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	AbsFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

func NegFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	NegFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

func SqrtFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	SqrtFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

func ReluFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	ReluFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&src[0]), len(dst))
}

var (
	addF32Funcs = []f32BinaryKernelImpl{
		{AddF32NEON, "neon", true},
		{AddF32Generic, "generic", true},
	}
	subF32Funcs = []f32BinaryKernelImpl{
		{SubF32NEON, "neon", true},
		{SubF32Generic, "generic", true},
	}
	mulF32Funcs = []f32BinaryKernelImpl{
		{MulF32NEON, "neon", true},
		{MulF32Generic, "generic", true},
	}
	divF32Funcs = []f32BinaryKernelImpl{
		{DivF32NEON, "neon", true},
		{DivF32Generic, "generic", true},
	}
	maxF32Funcs = []f32BinaryKernelImpl{
		{MaxF32NEON, "neon", true},
		{MaxF32Generic, "generic", true},
	}
	minF32Funcs = []f32BinaryKernelImpl{
		{MinF32NEON, "neon", true},
		{MinF32Generic, "generic", true},
	}
	absF32Funcs = []f32UnaryKernelImpl{
		{AbsF32NEON, "neon", true},
		{AbsF32Generic, "generic", true},
	}
	negF32Funcs = []f32UnaryKernelImpl{
		{NegF32NEON, "neon", true},
		{NegF32Generic, "generic", true},
	}
	sqrtF32Funcs = []f32UnaryKernelImpl{
		{SqrtF32NEON, "neon", true},
		{SqrtF32Generic, "generic", true},
	}
	reluF32Funcs = []f32UnaryKernelImpl{
		{ReluF32NEON, "neon", true},
		{ReluF32Generic, "generic", true},
	}
	axpyF32Funcs = []f32AxpyKernelImpl{
		{AxpyF32NEON, "neon", true},
		{AxpyF32Generic, "generic", true},
	}
	addF64Funcs = []f64BinaryKernelImpl{
		{AddF64NEON, "neon", true},
		{AddF64Generic, "generic", true},
	}
)
