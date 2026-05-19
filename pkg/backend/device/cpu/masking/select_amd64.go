//go:build amd64

package masking

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

var applyMaskF32Funcs = []f32ApplyMaskKernelImpl{
	{applyMaskF32AVX512, "avx512", cpu.X86.HasAVX512F},
	{applyMaskF32GenericKernel, "generic", true},
}

var causalMaskF32Funcs = []f32CausalMaskKernelImpl{
	{causalMaskF32AVX512, "avx512", cpu.X86.HasAVX512F},
	{causalMaskF32GenericKernel, "generic", true},
}

var alibiBiasF32Funcs = []f32ALiBiBiasKernelImpl{
	{alibiBiasF32AVX512, "avx512", cpu.X86.HasAVX512F},
	{alibiBiasF32GenericKernel, "generic", true},
}

func applyMaskF32AVX512(input, mask, output *float32, count int) {
	ApplyMaskF32AVX512(input, mask, output, count)
}

func causalMaskF32AVX512(output *float32, seqQ, seqK int) {
	CausalMaskF32AVX512(output, seqQ, seqK)
}

func alibiBiasF32AVX512(scores, slope, output *float32, seqQ, seqK int) {
	ALiBiBiasF32AVX512(scores, slope, output, seqQ, seqK)
}

func applyMaskF32GenericKernel(input, mask, output *float32, count int) {
	applyMaskF32Generic(
		unsafe.Pointer(input),
		unsafe.Pointer(mask),
		unsafe.Pointer(output),
		count,
	)
}

func causalMaskF32GenericKernel(output *float32, seqQ, seqK int) {
	causalMaskF32Generic(unsafe.Pointer(output), seqQ, seqK)
}

func alibiBiasF32GenericKernel(scores, slope, output *float32, seqQ, seqK int) {
	alibiBiasF32Generic(
		unsafe.Pointer(scores),
		unsafe.Pointer(slope),
		unsafe.Pointer(output),
		seqQ,
		seqK,
	)
}
