package masking

var applyMaskF32Kernel = func() func(input, mask, output *float32, count int) {
	return pickF32ApplyMaskKernel(applyMaskF32Funcs)
}()

var causalMaskF32Kernel = func() func(output *float32, seqQ, seqK int) {
	return pickF32CausalMaskKernel(causalMaskF32Funcs)
}()

var alibiBiasF32Kernel = func() func(scores, slope, output *float32, seqQ, seqK int) {
	return pickF32ALiBiBiasKernel(alibiBiasF32Funcs)
}()
