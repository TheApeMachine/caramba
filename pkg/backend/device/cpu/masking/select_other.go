//go:build !amd64

package masking

import "unsafe"

var applyMaskF32Funcs = []f32ApplyMaskKernelImpl{
	{applyMaskF32GenericKernel, "generic", true},
}

var causalMaskF32Funcs = []f32CausalMaskKernelImpl{
	{causalMaskF32GenericKernel, "generic", true},
}

var alibiBiasF32Funcs = []f32ALiBiBiasKernelImpl{
	{alibiBiasF32GenericKernel, "generic", true},
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
