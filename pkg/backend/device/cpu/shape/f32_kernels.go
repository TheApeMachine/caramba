package shape

var copyContiguousF32Kernel = func() func(dst, src *float32, count int) {
	return pickF32CopyContiguousKernel(copyContiguousF32Funcs)
}()

var whereF32Kernel = func() func(dst, positive, negative *float32, mask []byte, count int) {
	return pickF32WhereKernel(whereF32Funcs)
}()

var maskedFillF32Kernel = func() func(dst, input *float32, fill float32, mask []byte, count int) {
	return pickF32MaskedFillKernel(maskedFillF32Funcs)
}()
