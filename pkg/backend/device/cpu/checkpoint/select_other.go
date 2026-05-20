//go:build !amd64

package checkpoint

var encodeFloat32DataFuncs = []float32DataEncodeKernelImpl{
	{encodeFloat32DataScalar, "generic", true},
}

var decodeFloat32DataFuncs = []float32DataDecodeKernelImpl{
	{decodeFloat32DataScalar, "generic", true},
}
