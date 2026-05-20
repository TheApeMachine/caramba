//go:build !amd64

package model_editing

var weightGraftAddFloat32Funcs = []weightGraftAddKernelImpl{
	{weightGraftAddFloat32Scalar, "generic", true},
}

func weightGraftAddFloat32Scalar(weights, injection []float32, count int) {
	WeightGraftAddFloat32Scalar(weights[:count], injection[:count])
}
