package checkpoint

var encodeFloat32DataKernel = func() func(dst []byte, src []float32) {
	return pickEncodeFloat32DataKernel(encodeFloat32DataFuncs)
}()

var decodeFloat32DataKernel = func() func(dst []float32, src []byte) {
	return pickDecodeFloat32DataKernel(decodeFloat32DataFuncs)
}()
