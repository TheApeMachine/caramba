package kernels

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Mixed-precision dispatch for groupnorm, instancenorm, batchnorm_eval.
All follow the widen-bf16/fp16→f32 → run f32 norm → narrow pattern,
with f32 accumulation per §5.5.
*/

func init() {
	for _, paramDType := range []dtype.DType{dtype.BFloat16, dtype.Float16} {
		paramDType := paramDType

		// groupnorm: input, scale, bias, output
		Default.Register(Kernel{
			Name: "groupnorm",
			Signature: Signature{
				Layout:  tensor.LayoutDense,
				Inputs:  []dtype.DType{paramDType, paramDType, paramDType},
				Outputs: []dtype.DType{paramDType},
			},
			Locations: []tensor.Location{tensor.Host},
			Run: func(args ...tensor.Tensor) error {
				return runGroupNormMixed(args, paramDType)
			},
		})

		// instancenorm: input, scale, bias, output
		Default.Register(Kernel{
			Name: "instancenorm",
			Signature: Signature{
				Layout:  tensor.LayoutDense,
				Inputs:  []dtype.DType{paramDType, paramDType, paramDType},
				Outputs: []dtype.DType{paramDType},
			},
			Locations: []tensor.Location{tensor.Host},
			Run: func(args ...tensor.Tensor) error {
				return runInstanceNormMixed(args, paramDType)
			},
		})

		// batchnorm_eval: input, scale, bias, mean, variance, output
		Default.Register(Kernel{
			Name: "batchnorm_eval",
			Signature: Signature{
				Layout: tensor.LayoutDense,
				Inputs: []dtype.DType{
					paramDType, paramDType, paramDType, paramDType, paramDType,
				},
				Outputs: []dtype.DType{paramDType},
			},
			Locations: []tensor.Location{tensor.Host},
			Run: func(args ...tensor.Tensor) error {
				return runBatchNormEvalMixed(args, paramDType)
			},
		})
	}
}

func widenToF32(arg tensor.Tensor, kind dtype.DType, dst []float32) error {
	switch kind {
	case dtype.BFloat16:
		view, err := arg.BFloat16Native()
		if err != nil {
			return err
		}
		if len(view) != len(dst) {
			return tensor.ErrShapeMismatch
		}
		bfloat16BulkToFloat32(dst, view)
	case dtype.Float16:
		view, err := arg.Float16Native()
		if err != nil {
			return err
		}
		if len(view) != len(dst) {
			return tensor.ErrShapeMismatch
		}
		float16BulkToFloat32(dst, view)
	}
	return nil
}

func narrowFromF32(arg tensor.Tensor, kind dtype.DType, src []float32) error {
	switch kind {
	case dtype.BFloat16:
		view, err := arg.BFloat16Native()
		if err != nil {
			return err
		}
		if len(view) != len(src) {
			return tensor.ErrShapeMismatch
		}
		float32BulkToBFloat16(view, src)
	case dtype.Float16:
		view, err := arg.Float16Native()
		if err != nil {
			return err
		}
		if len(view) != len(src) {
			return tensor.ErrShapeMismatch
		}
		float32BulkToFloat16(view, src)
	}
	return nil
}

func argLen(arg tensor.Tensor, kind dtype.DType) (int, error) {
	switch kind {
	case dtype.BFloat16:
		view, err := arg.BFloat16Native()
		if err != nil {
			return 0, err
		}
		return len(view), nil
	case dtype.Float16:
		view, err := arg.Float16Native()
		if err != nil {
			return 0, err
		}
		return len(view), nil
	}
	return 0, tensor.ErrShapeMismatch
}

func runGroupNormMixed(args []tensor.Tensor, kind dtype.DType) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	dims := args[0].Shape().Dims()
	if len(dims) != 3 {
		return tensor.ErrShapeMismatch
	}

	batch := dims[0]
	channels := dims[1]
	spatial := dims[2]

	config := DefaultGroupNormConfig()
	if channels%config.Groups != 0 {
		return tensor.ErrShapeMismatch
	}

	inputLen, err := argLen(args[0], kind)
	if err != nil {
		return err
	}

	inputF32 := borrowFloat32Buffer(inputLen)
	scaleF32 := borrowFloat32Buffer(channels)
	biasF32 := borrowFloat32Buffer(channels)
	outF32 := borrowFloat32Buffer(inputLen)

	defer releaseFloat32Buffer(inputF32)
	defer releaseFloat32Buffer(scaleF32)
	defer releaseFloat32Buffer(biasF32)
	defer releaseFloat32Buffer(outF32)

	if err := widenToF32(args[0], kind, inputF32); err != nil {
		return err
	}
	if err := widenToF32(args[1], kind, scaleF32); err != nil {
		return err
	}
	if err := widenToF32(args[2], kind, biasF32); err != nil {
		return err
	}

	groupNormSlices(config, inputF32, scaleF32, biasF32, outF32, batch, channels, spatial)

	return narrowFromF32(args[3], kind, outF32)
}

func runInstanceNormMixed(args []tensor.Tensor, kind dtype.DType) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	dims := args[0].Shape().Dims()
	if len(dims) != 3 {
		return tensor.ErrShapeMismatch
	}

	batch := dims[0]
	channels := dims[1]
	spatial := dims[2]

	inputLen, err := argLen(args[0], kind)
	if err != nil {
		return err
	}

	inputF32 := borrowFloat32Buffer(inputLen)
	scaleF32 := borrowFloat32Buffer(channels)
	biasF32 := borrowFloat32Buffer(channels)
	outF32 := borrowFloat32Buffer(inputLen)

	defer releaseFloat32Buffer(inputF32)
	defer releaseFloat32Buffer(scaleF32)
	defer releaseFloat32Buffer(biasF32)
	defer releaseFloat32Buffer(outF32)

	if err := widenToF32(args[0], kind, inputF32); err != nil {
		return err
	}
	if err := widenToF32(args[1], kind, scaleF32); err != nil {
		return err
	}
	if err := widenToF32(args[2], kind, biasF32); err != nil {
		return err
	}

	instanceNormSlices(inputF32, scaleF32, biasF32, outF32, batch, channels, spatial)

	return narrowFromF32(args[3], kind, outF32)
}

func runBatchNormEvalMixed(args []tensor.Tensor, kind dtype.DType) error {
	if len(args) != 6 {
		return tensor.ErrShapeMismatch
	}

	dims := args[0].Shape().Dims()
	if len(dims) != 3 {
		return tensor.ErrShapeMismatch
	}

	batch := dims[0]
	channels := dims[1]
	spatial := dims[2]

	inputLen, err := argLen(args[0], kind)
	if err != nil {
		return err
	}

	inputF32 := borrowFloat32Buffer(inputLen)
	scaleF32 := borrowFloat32Buffer(channels)
	biasF32 := borrowFloat32Buffer(channels)
	meanF32 := borrowFloat32Buffer(channels)
	varF32 := borrowFloat32Buffer(channels)
	outF32 := borrowFloat32Buffer(inputLen)

	defer releaseFloat32Buffer(inputF32)
	defer releaseFloat32Buffer(scaleF32)
	defer releaseFloat32Buffer(biasF32)
	defer releaseFloat32Buffer(meanF32)
	defer releaseFloat32Buffer(varF32)
	defer releaseFloat32Buffer(outF32)

	if err := widenToF32(args[0], kind, inputF32); err != nil {
		return err
	}
	if err := widenToF32(args[1], kind, scaleF32); err != nil {
		return err
	}
	if err := widenToF32(args[2], kind, biasF32); err != nil {
		return err
	}
	if err := widenToF32(args[3], kind, meanF32); err != nil {
		return err
	}
	if err := widenToF32(args[4], kind, varF32); err != nil {
		return err
	}

	batchNormEvalSlices(inputF32, scaleF32, biasF32, meanF32, varF32, outF32, batch, channels, spatial)

	return narrowFromF32(args[5], kind, outF32)
}
