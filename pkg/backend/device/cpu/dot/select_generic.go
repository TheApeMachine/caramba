//go:build !386 && !amd64 && !arm64

package dot

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func DotFloat32Native(left, right []float32) float32 {
	if len(left) == 0 {
		return 0
	}

	return Dot(
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(left),
		dtype.Float32,
	)
}

func DotBFloat16Native(left, right []dtype.BF16) dtype.BF16 {
	if len(left) == 0 {
		return 0
	}

	return dtype.NewBfloat16FromFloat32(Dot(
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(left),
		dtype.BFloat16,
	))
}

func DotFloat16Native(left, right []dtype.F16) dtype.F16 {
	if len(left) == 0 {
		return 0
	}

	return dtype.Fromfloat32(Dot(
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(left),
		dtype.Float16,
	))
}

func DotInt8Native(left, right []int8) int32 {
	if len(left) == 0 {
		return 0
	}

	return int32(Dot(
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(left),
		dtype.Int8,
	))
}

var (
	dotF32Funcs = []f32DotKernelImpl{
		{DotF32Generic, "generic", true},
	}
	dotBF16Funcs = []bf16DotKernelImpl{
		{DotBF16Generic, "generic", true},
	}
	dotFP16Funcs = []fp16DotKernelImpl{
		{DotFP16Generic, "generic", true},
	}
	dotInt8Funcs = []int8DotKernelImpl{
		{DotInt8Generic, "generic", true},
	}
)
