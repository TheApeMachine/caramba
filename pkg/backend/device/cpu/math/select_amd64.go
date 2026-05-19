//go:build amd64

package math

import "golang.org/x/sys/cpu"

var invSqrtDimScaleF32Funcs = []f32InvSqrtDimScaleKernelImpl{
	{InvSqrtDimScaleF32AVX512, "avx512", cpu.X86.HasAVX512F},
	{InvSqrtDimScaleGeneric, "generic", true},
}

var logSumExpF32Funcs = []f32LogSumExpKernelImpl{
	{LogSumExpF32AVX512, "avx512", cpu.X86.HasAVX512F},
	{LogSumExpGeneric, "generic", true},
}

var outerF32Funcs = []f32OuterKernelImpl{
	{OuterF32AVX512, "avx512", cpu.X86.HasAVX512F},
	{OuterGeneric, "generic", true},
}
