package tokenizer

var packInt32Kernel = func() func(dst, src *int32, count int) {
	return pickInt32PackKernel(packInt32Funcs)
}()
