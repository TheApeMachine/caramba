//go:build amd64

package reduction

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
	"golang.org/x/sys/cpu"
)

func SumFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	return Sum(
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.Float32,
	)
}

func SumBFloat16Native(values []dtype.BF16) dtype.BF16 {
	if len(values) == 0 {
		return 0
	}

	return dtype.NewBfloat16FromFloat32(Sum(
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.BFloat16,
	))
}

func SumFloat16Native(values []dtype.F16) dtype.F16 {
	if len(values) == 0 {
		return 0
	}

	return dtype.Fromfloat32(Sum(
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.Float16,
	))
}

func ReduceProdFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	return Prod(
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.Float32,
	)
}

func ReduceMinFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	return ReduceMin(
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.Float32,
	)
}

func ReduceMaxFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	return ReduceMax(
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.Float32,
	)
}

func L1NormFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	return L1Norm(
		unsafe.Pointer(&values[0]),
		len(values),
		dtype.Float32,
	)
}

var (
	sumF32Funcs = []f32ReduceKernelImpl{
		{SumF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{SumF32Generic, "generic", true},
	}
	prodF32Funcs = []f32ReduceKernelImpl{
		{ProdF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{ProdF32Generic, "generic", true},
	}
	minF32Funcs = []f32ReduceKernelImpl{
		{MinF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{MinF32Generic, "generic", true},
	}
	maxF32Funcs = []f32ReduceKernelImpl{
		{MaxF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{MaxF32Generic, "generic", true},
	}
	l1NormF32Funcs = []f32ReduceKernelImpl{
		{L1NormF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{L1NormF32Generic, "generic", true},
	}
	sumBF16Funcs = []bf16SumKernelImpl{
		{SumBF16Generic, "generic", true},
	}
	sumFP16Funcs = []fp16SumKernelImpl{
		{SumFP16Generic, "generic", true},
	}
)
