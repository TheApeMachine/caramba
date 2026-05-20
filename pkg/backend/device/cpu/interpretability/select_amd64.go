//go:build amd64

package interpretability

import "golang.org/x/sys/cpu"

var activationSteerFloat32Funcs = []activationSteerKernelImpl{
	{activationSteerFloat32AVX512, "avx512", cpu.X86.HasAVX512F},
	{activationSteerFloat32Scalar, "generic", true},
}

func activationSteerFloat32AVX512(
	destination, base, direction []float32,
	coefficient float32,
	count int,
) {
	ActivationSteerFloat32AVX512(
		&destination[0], &base[0], &direction[0], coefficient, count,
	)
}

func activationSteerFloat32Scalar(
	destination, base, direction []float32,
	coefficient float32,
	count int,
) {
	ActivationSteerFloat32Scalar(destination[:count], base[:count], direction[:count], coefficient)
}
