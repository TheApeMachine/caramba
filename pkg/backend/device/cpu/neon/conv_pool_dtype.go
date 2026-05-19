package neon

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Mixed-precision dispatch for conv1d/conv2d/conv3d/conv_transpose2d
and the pooling family (max_pool2d, avg_pool2d, adaptive_*). All
follow the widen→f32 → run existing f32 kernel via constructed temp
tensors → narrow pattern.

The temp-tensor approach keeps the f32 math single-source (it lives
in the original *Float32 runners) at the cost of one allocation per
call. Future SIMD work can replace this with direct slice-level
dispatch.
*/

func init() {
	// All-input-all-output paramDType signatures.
	for _, paramDType := range []dtype.DType{dtype.BFloat16, dtype.Float16} {
		paramDType := paramDType

		registerConvPoolMixed("conv1d", paramDType, 4, runConv1DDefault)
		registerConvPoolMixed("conv2d", paramDType, 4, runConv2DDefault)
		registerConvPoolMixed("conv3d", paramDType, 4, runConv3DDefault)
		registerConvPoolMixed("conv_transpose2d", paramDType, 4, runConvTranspose2DDefault)
		registerConvPoolMixed("max_pool2d", paramDType, 2, runMaxPool2DDefault)
		registerConvPoolMixed("avg_pool2d", paramDType, 2, runAvgPool2DDefault)
		registerConvPoolMixed("adaptive_max_pool2d", paramDType, 2, runAdaptiveMaxPool2D)
		registerConvPoolMixed("adaptive_avg_pool2d", paramDType, 2, runAdaptiveAvgPool2D)
	}
}

func registerConvPoolMixed(
	name string, paramDType dtype.DType, arity int,
	f32Runner func(args ...tensor.Tensor) error,
) {
	inputDtypes := make([]dtype.DType, arity-1)

	for index := range inputDtypes {
		inputDtypes[index] = paramDType
	}

	Default.Register(Kernel{
		Name: name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  inputDtypes,
			Outputs: []dtype.DType{paramDType},
		},
		Locations: []tensor.Location{tensor.Host},
		Run: func(args ...tensor.Tensor) error {
			return runConvPoolMixed(args, paramDType, f32Runner)
		},
	})
}

func runConvPoolMixed(
	args []tensor.Tensor,
	kind dtype.DType,
	f32Runner func(args ...tensor.Tensor) error,
) error {
	// Allocate temporary f32 tensors for every input and the output,
	// matching their shapes. Widen input tensors, run the f32 kernel,
	// narrow the output back.
	temps := make([]tensor.Tensor, len(args))

	for index, arg := range args {
		shape := arg.Shape()
		temp, err := tensor.NewZeroed(shape, dtype.Float32)

		if err != nil {
			return err
		}

		temps[index] = temp

		// Widen inputs into temp (output temp stays zero-initialized).
		if index < len(args)-1 {
			tempView, _ := temp.Float32Native()

			if err := widenToF32(arg, kind, tempView); err != nil {
				return err
			}
		}
	}

	if err := f32Runner(temps...); err != nil {
		return err
	}

	// Narrow the f32 output back into the caller's dtype tensor.
	outTempView, _ := temps[len(args)-1].Float32Native()
	return narrowFromF32(args[len(args)-1], kind, outTempView)
}
