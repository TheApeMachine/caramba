package neon

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Additional normalization kernels beyond LayerNorm/RMSNorm: GroupNorm,
BatchNorm (eval-mode), InstanceNorm. All operate on NCHW-flavored
tensors but the implementations here accept the simpler [N, C, S]
flattened shape — the orchestrator flattens spatial dims before
dispatching.

Args order for groupnorm: (input, scale, bias, output).
Args order for batchnorm_eval: (input, scale, bias, mean, variance, output).
Args order for instancenorm: (input, scale, bias, output).
*/

const normEpsilon = 1e-5

type GroupNormConfig struct {
	Groups int
}

func DefaultGroupNormConfig() GroupNormConfig {
	return GroupNormConfig{Groups: 32}
}

func init() {
	Default.Register(Kernel{
		Name: "groupnorm",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runGroupNormDefault,
	})

	Default.Register(Kernel{
		Name: "instancenorm",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runInstanceNormFloat32,
	})

	Default.Register(Kernel{
		Name: "batchnorm_eval",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runBatchNormEvalFloat32,
	})
}

func runGroupNormDefault(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return GroupNormFloat32(DefaultGroupNormConfig(), args[0], args[1], args[2], args[3])
}

/*
GroupNormFloat32 operates on [batch, channels, spatial] tensors with
channels split into config.Groups groups. Scale and bias have length
channels.
*/
func GroupNormFloat32(
	config GroupNormConfig,
	input, scale, bias, out tensor.Tensor,
) error {
	dims := input.Shape().Dims()

	if len(dims) != 3 {
		return tensor.ErrShapeMismatch
	}

	batch := dims[0]
	channels := dims[1]
	spatial := dims[2]

	if channels%config.Groups != 0 {
		return tensor.ErrShapeMismatch
	}

	channelsPerGroup := channels / config.Groups
	groupSize := channelsPerGroup * spatial

	inputView, err := input.Float32Native()

	if err != nil {
		return err
	}

	scaleView, err := scale.Float32Native()

	if err != nil {
		return err
	}

	biasView, err := bias.Float32Native()

	if err != nil {
		return err
	}

	outView, err := out.Float32Native()

	if err != nil {
		return err
	}

	if len(scaleView) != channels || len(biasView) != channels ||
		len(outView) != len(inputView) {
		return tensor.ErrShapeMismatch
	}

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for groupIndex := 0; groupIndex < config.Groups; groupIndex++ {
			channelStart := groupIndex * channelsPerGroup
			groupStart := batchIndex*channels*spatial + channelStart*spatial

			normalizeGroup(
				inputView[groupStart:groupStart+groupSize],
				outView[groupStart:groupStart+groupSize],
				scaleView[channelStart:channelStart+channelsPerGroup],
				biasView[channelStart:channelStart+channelsPerGroup],
				channelsPerGroup,
				spatial,
			)
		}
	}

	return nil
}

func normalizeGroup(
	inputSlice, outSlice, scaleSlice, biasSlice []float32,
	channelsPerGroup, spatial int,
) {
	var sum float64

	for _, value := range inputSlice {
		sum += float64(value)
	}

	mean := sum / float64(len(inputSlice))

	var variance float64

	for _, value := range inputSlice {
		delta := float64(value) - mean
		variance += delta * delta
	}

	variance /= float64(len(inputSlice))
	invStdDev := 1.0 / math.Sqrt(variance+normEpsilon)

	for channelIndex := 0; channelIndex < channelsPerGroup; channelIndex++ {
		for spatialIndex := 0; spatialIndex < spatial; spatialIndex++ {
			localIndex := channelIndex*spatial + spatialIndex
			value := float64(inputSlice[localIndex])
			normalized := (value - mean) * invStdDev
			outSlice[localIndex] = float32(normalized)*scaleSlice[channelIndex] + biasSlice[channelIndex]
		}
	}
}

/*
InstanceNorm normalizes each (batch, channel) pair independently
across the spatial dimension.
*/
func runInstanceNormFloat32(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	input, scale, bias, out := args[0], args[1], args[2], args[3]

	dims := input.Shape().Dims()

	if len(dims) != 3 {
		return tensor.ErrShapeMismatch
	}

	batch := dims[0]
	channels := dims[1]
	spatial := dims[2]

	inputView, _ := input.Float32Native()
	scaleView, _ := scale.Float32Native()
	biasView, _ := bias.Float32Native()
	outView, _ := out.Float32Native()

	if len(scaleView) != channels || len(biasView) != channels ||
		len(outView) != len(inputView) {
		return tensor.ErrShapeMismatch
	}

	instanceNormSlices(inputView, scaleView, biasView, outView, batch, channels, spatial)
	return nil
}

func instanceNormSlices(input, scale, bias, output []float32, batch, channels, spatial int) {
	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for channelIndex := 0; channelIndex < channels; channelIndex++ {
			start := (batchIndex*channels + channelIndex) * spatial
			row := input[start : start+spatial]
			outRow := output[start : start+spatial]

			var sum float64
			for _, value := range row {
				sum += float64(value)
			}
			mean := sum / float64(spatial)

			var variance float64
			for _, value := range row {
				delta := float64(value) - mean
				variance += delta * delta
			}
			variance /= float64(spatial)
			invStdDev := 1.0 / math.Sqrt(variance+normEpsilon)

			for spatialIndex, value := range row {
				normalized := (float64(value) - mean) * invStdDev
				outRow[spatialIndex] = float32(normalized)*scale[channelIndex] + bias[channelIndex]
			}
		}
	}
}

/*
BatchNormEval applies the inference-mode batch normalization: each
channel is normalized by precomputed running mean/variance, then
scaled+biased.
*/
func runBatchNormEvalFloat32(args ...tensor.Tensor) error {
	if len(args) != 6 {
		return tensor.ErrShapeMismatch
	}

	input, _ := args[0].Float32Native()
	scale, _ := args[1].Float32Native()
	bias, _ := args[2].Float32Native()
	mean, _ := args[3].Float32Native()
	variance, _ := args[4].Float32Native()
	out, _ := args[5].Float32Native()

	dims := args[0].Shape().Dims()

	if len(dims) != 3 {
		return tensor.ErrShapeMismatch
	}

	batch := dims[0]
	channels := dims[1]
	spatial := dims[2]

	if len(scale) != channels || len(bias) != channels ||
		len(mean) != channels || len(variance) != channels ||
		len(out) != len(input) {
		return tensor.ErrShapeMismatch
	}

	batchNormEvalSlices(input, scale, bias, mean, variance, out, batch, channels, spatial)
	return nil
}

func batchNormEvalSlices(
	input, scale, bias, mean, variance, output []float32,
	batch, channels, spatial int,
) {
	for channelIndex := 0; channelIndex < channels; channelIndex++ {
		invStdDev := 1.0 / float32(math.Sqrt(float64(variance[channelIndex])+normEpsilon))

		for batchIndex := 0; batchIndex < batch; batchIndex++ {
			start := (batchIndex*channels + channelIndex) * spatial

			for spatialIndex := 0; spatialIndex < spatial; spatialIndex++ {
				value := input[start+spatialIndex]
				normalized := (value - mean[channelIndex]) * invStdDev
				output[start+spatialIndex] = normalized*scale[channelIndex] + bias[channelIndex]
			}
		}
	}
}

// groupNormSlices wraps GroupNormFloat32's per-group work so the
// mixed-precision wrappers can dispatch without rebuilding tensor
// wrappers around scratch buffers.
func groupNormSlices(
	config GroupNormConfig,
	input, scale, bias, output []float32,
	batch, channels, spatial int,
) {
	channelsPerGroup := channels / config.Groups
	groupSize := channelsPerGroup * spatial

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for groupIndex := 0; groupIndex < config.Groups; groupIndex++ {
			channelStart := groupIndex * channelsPerGroup
			groupStart := batchIndex*channels*spatial + channelStart*spatial

			normalizeGroup(
				input[groupStart:groupStart+groupSize],
				output[groupStart:groupStart+groupSize],
				scale[channelStart:channelStart+channelsPerGroup],
				bias[channelStart:channelStart+channelsPerGroup],
				channelsPerGroup,
				spatial,
			)
		}
	}
}
