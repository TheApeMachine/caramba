package kernels

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Mixed-precision optimizer dispatchers. Standard AMP/bf16 training
convention: params, gradients, and output stored at the reduced dtype
(bf16 or fp16); all optimizer state (momentum, variance, etc.) stays
at f32 for numerical stability across the long-tail of training steps.

Per-optimizer kernels in this file widen params and gradients to f32
scratch buffers via NEON-accelerated bulk conversion, call the
slice-based f32 update math, then narrow the output back. State
tensors pass through as f32 unchanged.
*/

// optimizerSliceFn updates output (and the in-place state slices) from
// params, gradients, and the per-optimizer state slices.
type optimizerSliceFn func(params, gradients []float32, state [][]float32, output []float32)

// runMixedOptimizerBFloat16 dispatches an optimizer step for the
// standard AMP signature: args[0] params (bf16), args[1] gradients
// (bf16), args[2..2+stateCount-1] state tensors (f32), args[last]
// output (bf16).
func runMixedOptimizerBFloat16(args []tensor.Tensor, stateCount int, step optimizerSliceFn) error {
	if len(args) != stateCount+3 {
		return tensor.ErrShapeMismatch
	}

	paramsBF16, err := args[0].BFloat16Native()

	if err != nil {
		return err
	}

	gradBF16, err := args[1].BFloat16Native()

	if err != nil {
		return err
	}

	outBF16, err := args[stateCount+2].BFloat16Native()

	if err != nil {
		return err
	}

	n := len(paramsBF16)

	if len(gradBF16) != n || len(outBF16) != n {
		return tensor.ErrShapeMismatch
	}

	stateSlices := make([][]float32, stateCount)

	for index := range stateCount {
		state, err := args[2+index].Float32Native()

		if err != nil {
			return err
		}

		if len(state) != n {
			return tensor.ErrShapeMismatch
		}

		stateSlices[index] = state
	}

	paramsF32 := borrowFloat32Buffer(n)
	gradF32 := borrowFloat32Buffer(n)
	outF32 := borrowFloat32Buffer(n)

	defer releaseFloat32Buffer(paramsF32)
	defer releaseFloat32Buffer(gradF32)
	defer releaseFloat32Buffer(outF32)

	bfloat16BulkToFloat32(paramsF32, paramsBF16)
	bfloat16BulkToFloat32(gradF32, gradBF16)

	step(paramsF32, gradF32, stateSlices, outF32)

	float32BulkToBFloat16(outBF16, outF32)
	return nil
}

// runMixedOptimizerFloat16 mirrors runMixedOptimizerBFloat16 but with
// fp16 params/gradients/output.
func runMixedOptimizerFloat16(args []tensor.Tensor, stateCount int, step optimizerSliceFn) error {
	if len(args) != stateCount+3 {
		return tensor.ErrShapeMismatch
	}

	paramsF16, err := args[0].Float16Native()

	if err != nil {
		return err
	}

	gradF16, err := args[1].Float16Native()

	if err != nil {
		return err
	}

	outF16, err := args[stateCount+2].Float16Native()

	if err != nil {
		return err
	}

	n := len(paramsF16)

	if len(gradF16) != n || len(outF16) != n {
		return tensor.ErrShapeMismatch
	}

	stateSlices := make([][]float32, stateCount)

	for index := range stateCount {
		state, err := args[2+index].Float32Native()

		if err != nil {
			return err
		}

		if len(state) != n {
			return tensor.ErrShapeMismatch
		}

		stateSlices[index] = state
	}

	paramsF32 := borrowFloat32Buffer(n)
	gradF32 := borrowFloat32Buffer(n)
	outF32 := borrowFloat32Buffer(n)

	defer releaseFloat32Buffer(paramsF32)
	defer releaseFloat32Buffer(gradF32)
	defer releaseFloat32Buffer(outF32)

	float16BulkToFloat32(paramsF32, paramsF16)
	float16BulkToFloat32(gradF32, gradF16)

	step(paramsF32, gradF32, stateSlices, outF32)

	float32BulkToFloat16(outF16, outF32)
	return nil
}

// registerMixedOptimizer registers bf16 + fp16 signatures for an
// optimizer with the given name and number of f32 state tensors.
func registerMixedOptimizer(name string, stateCount int, step optimizerSliceFn) {
	inputDtypesBF16 := make([]dtype.DType, 2+stateCount)
	inputDtypesBF16[0] = dtype.BFloat16
	inputDtypesBF16[1] = dtype.BFloat16

	inputDtypesF16 := make([]dtype.DType, 2+stateCount)
	inputDtypesF16[0] = dtype.Float16
	inputDtypesF16[1] = dtype.Float16

	for index := range stateCount {
		inputDtypesBF16[2+index] = dtype.Float32
		inputDtypesF16[2+index] = dtype.Float32
	}

	bfRun := func(args ...tensor.Tensor) error {
		return runMixedOptimizerBFloat16(args, stateCount, step)
	}

	f16Run := func(args ...tensor.Tensor) error {
		return runMixedOptimizerFloat16(args, stateCount, step)
	}

	Default.Register(Kernel{
		Name: name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  inputDtypesBF16,
			Outputs: []dtype.DType{dtype.BFloat16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       bfRun,
	})

	Default.Register(Kernel{
		Name: name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  inputDtypesF16,
			Outputs: []dtype.DType{dtype.Float16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       f16Run,
	})
}

func init() {
	// Adam: 2 states (firstMoment, secondMoment)
	registerMixedOptimizer("adam_step", 2, func(params, gradients []float32, state [][]float32, output []float32) {
		adamStepSlices(DefaultAdamConfig(), params, gradients, state[0], state[1], output)
	})

	// AdamW: 2 states
	registerMixedOptimizer("adamw_step", 2, func(params, gradients []float32, state [][]float32, output []float32) {
		adamWStepSlices(DefaultAdamWConfig(), params, gradients, state[0], state[1], output)
	})

	// Lion: 1 state (momentum)
	registerMixedOptimizer("lion_step", 1, func(params, gradients []float32, state [][]float32, output []float32) {
		lionStepSlices(DefaultLionConfig(), params, gradients, state[0], output)
	})

	// SGD with momentum: 1 state
	registerMixedOptimizer("sgd_step", 1, func(params, gradients []float32, state [][]float32, output []float32) {
		sgdStepSlices(DefaultSGDConfig(), params, gradients, state[0], output)
	})

	// Adamax: 2 states (firstMoment, infinityMoment)
	registerMixedOptimizer("adamax_step", 2, func(params, gradients []float32, state [][]float32, output []float32) {
		adamaxStepSlices(DefaultAdamaxConfig(), params, gradients, state[0], state[1], output)
	})

	// Adagrad: 1 state (accumulator)
	registerMixedOptimizer("adagrad_step", 1, func(params, gradients []float32, state [][]float32, output []float32) {
		adagradStepSlices(DefaultAdagradConfig(), params, gradients, state[0], output)
	})

	// RMSprop: 1 state (secondMoment)
	registerMixedOptimizer("rmsprop_step", 1, func(params, gradients []float32, state [][]float32, output []float32) {
		rmspropStepSlices(DefaultRMSpropConfig(), params, gradients, state[0], output)
	})

	// LARS: 1 state (momentum)
	registerMixedOptimizer("lars_step", 1, func(params, gradients []float32, state [][]float32, output []float32) {
		larsStepSlices(DefaultLARSConfig(), params, gradients, state[0], output)
	})

	// LBFGS: 0 state (the host reference is gradient descent)
	registerMixedOptimizer("lbfgs_step", 0, func(params, gradients []float32, state [][]float32, output []float32) {
		lbfgsStepSlices(DefaultLBFGSConfig(), params, gradients, output)
	})

	// Hebbian has a unique signature (weights, post, pre, output) so
	// it's registered directly rather than via the generic helper.
	registerHebbianMixedDType()
}

func registerHebbianMixedDType() {
	for _, paramDType := range []dtype.DType{dtype.BFloat16, dtype.Float16} {
		paramDType := paramDType
		Default.Register(Kernel{
			Name: "hebbian_step",
			Signature: Signature{
				Layout:  tensor.LayoutDense,
				Inputs:  []dtype.DType{paramDType, paramDType, paramDType},
				Outputs: []dtype.DType{paramDType},
			},
			Locations: []tensor.Location{tensor.Host},
			Run: func(args ...tensor.Tensor) error {
				if paramDType == dtype.BFloat16 {
					return runHebbianBFloat16(args)
				}
				return runHebbianFloat16(args)
			},
		})
	}
}

func runHebbianBFloat16(args []tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	weightsBF16, err := args[0].BFloat16Native()

	if err != nil {
		return err
	}

	postBF16, err := args[1].BFloat16Native()

	if err != nil {
		return err
	}

	preBF16, err := args[2].BFloat16Native()

	if err != nil {
		return err
	}

	outBF16, err := args[3].BFloat16Native()

	if err != nil {
		return err
	}

	dims := args[0].Shape().Dims()

	if len(dims) != 2 || dims[0] != len(postBF16) || dims[1] != len(preBF16) ||
		len(outBF16) != len(weightsBF16) {
		return tensor.ErrShapeMismatch
	}

	weightsF32 := borrowFloat32Buffer(len(weightsBF16))
	postF32 := borrowFloat32Buffer(len(postBF16))
	preF32 := borrowFloat32Buffer(len(preBF16))
	outF32 := borrowFloat32Buffer(len(outBF16))

	defer releaseFloat32Buffer(weightsF32)
	defer releaseFloat32Buffer(postF32)
	defer releaseFloat32Buffer(preF32)
	defer releaseFloat32Buffer(outF32)

	bfloat16BulkToFloat32(weightsF32, weightsBF16)
	bfloat16BulkToFloat32(postF32, postBF16)
	bfloat16BulkToFloat32(preF32, preBF16)

	hebbianStepSlices(DefaultHebbianConfig(), weightsF32, postF32, preF32, outF32, dims[1])

	float32BulkToBFloat16(outBF16, outF32)
	return nil
}

func runHebbianFloat16(args []tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	weightsF16, err := args[0].Float16Native()

	if err != nil {
		return err
	}

	postF16, err := args[1].Float16Native()

	if err != nil {
		return err
	}

	preF16, err := args[2].Float16Native()

	if err != nil {
		return err
	}

	outF16, err := args[3].Float16Native()

	if err != nil {
		return err
	}

	dims := args[0].Shape().Dims()

	if len(dims) != 2 || dims[0] != len(postF16) || dims[1] != len(preF16) ||
		len(outF16) != len(weightsF16) {
		return tensor.ErrShapeMismatch
	}

	weightsF32 := borrowFloat32Buffer(len(weightsF16))
	postF32 := borrowFloat32Buffer(len(postF16))
	preF32 := borrowFloat32Buffer(len(preF16))
	outF32 := borrowFloat32Buffer(len(outF16))

	defer releaseFloat32Buffer(weightsF32)
	defer releaseFloat32Buffer(postF32)
	defer releaseFloat32Buffer(preF32)
	defer releaseFloat32Buffer(outF32)

	float16BulkToFloat32(weightsF32, weightsF16)
	float16BulkToFloat32(postF32, postF16)
	float16BulkToFloat32(preF32, preF16)

	hebbianStepSlices(DefaultHebbianConfig(), weightsF32, postF32, preF32, outF32, dims[1])

	float32BulkToFloat16(outF16, outF32)
	return nil
}
