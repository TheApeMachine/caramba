package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

var metalPhysicsDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

type metalPhysicsBinaryOp int

const (
	metalPhysicsLaplacian metalPhysicsBinaryOp = iota
	metalPhysicsLaplacian4
	metalPhysicsGrad1D
	metalPhysicsDivergence1D
	metalPhysicsQuantumPotential
	metalPhysicsBohmianVelocity
)

func init() {
	for _, storageDType := range metalPhysicsDTypes {
		registerMetalPhysicsKernels(storageDType)
	}
}

func registerMetalPhysicsKernels(storageDType dtype.DType) {
	registerMetalPhysicsBinaryKernel("laplacian", storageDType, metalPhysicsLaplacian)
	registerMetalPhysicsBinaryKernel("laplacian4", storageDType, metalPhysicsLaplacian4)
	registerMetalPhysicsBinaryKernel("grad1d", storageDType, metalPhysicsGrad1D)
	registerMetalPhysicsBinaryKernel("divergence1d", storageDType, metalPhysicsDivergence1D)
	registerMetalPhysicsFFTKernel("fft1d", storageDType, runMetalFFT1DKernel)
	registerMetalPhysicsFFTKernel("ifft1d", storageDType, runMetalIFFT1DKernel)
	registerMetalPhysicsBinaryKernel("quantum_potential", storageDType, metalPhysicsQuantumPotential)
	registerMetalPhysicsBinaryKernel("bohmian_velocity", storageDType, metalPhysicsBohmianVelocity)
	registerMetalMadelungContinuityKernel(storageDType)
}

func registerMetalPhysicsBinaryKernel(
	name string,
	storageDType dtype.DType,
	operation metalPhysicsBinaryOp,
) {
	kernels.Default.Register(kernels.Kernel{
		Name: name,
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run: func(args ...tensor.Tensor) error {
			return runMetalPhysicsBinaryKernel(operation, args...)
		},
	})
}

func registerMetalPhysicsFFTKernel(
	name string,
	storageDType dtype.DType,
	run func(...tensor.Tensor) error,
) {
	kernels.Default.Register(kernels.Kernel{
		Name: name,
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, storageDType},
			Outputs: []dtype.DType{storageDType, storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       run,
	})
}

func registerMetalMadelungContinuityKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "madelung_continuity",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType, storageDType, storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalMadelungContinuityKernel,
	})
}

func runMetalPhysicsBinaryKernel(
	operation metalPhysicsBinaryOp,
	args ...tensor.Tensor,
) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalPhysicsBinary(operation, args[0], args[1], args[2])
}

func runMetalFFT1DKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalFFT1D(args[0], args[1], args[2], args[3])
}

func runMetalIFFT1DKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalIFFT1D(args[0], args[1], args[2], args[3])
}

func runMetalMadelungContinuityKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalMadelungContinuity(args[0], args[1], args[2], args[3])
}
