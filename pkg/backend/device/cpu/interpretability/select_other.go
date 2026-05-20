//go:build !amd64

package interpretability

var activationSteerFloat32Funcs = []activationSteerKernelImpl{
	{activationSteerFloat32Scalar, "generic", true},
}

func activationSteerFloat32Scalar(
	destination, base, direction []float32,
	coefficient float32,
	count int,
) {
	ActivationSteerFloat32Scalar(destination[:count], base[:count], direction[:count], coefficient)
}
