package math

import (
	"fmt"
	gomath "math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
GroupNorm normalizes NCHW tensors over each batch/group channel partition.
The reductions run through the same architecture-specific SIMD primitives used
by the rest of the CPU math package.
*/
type GroupNorm struct{}

func NewGroupNorm() *GroupNorm {
	return &GroupNorm{}
}

func (groupNorm *GroupNorm) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("math.groupnorm"); err != nil {
		return nil, err
	}

	batch, channels, height, width, err := groupNormLayout(stateDict)

	if err != nil {
		return nil, err
	}

	stateDict.EnsureOperationOutLen(len(stateDict.Inputs[0]))
	groupNormKernel(
		stateDict.Out,
		stateDict.Inputs[0],
		stateDict.Weight,
		stateDict.Bias,
		stateDict.Eps,
		stateDict.Groups,
		batch,
		channels,
		height,
		width,
	)

	return stateDict, nil
}

func groupNormLayout(stateDict *state.Dict) (int, int, int, int, error) {
	shape := stateDict.OperationShape()

	if len(shape) != 4 {
		return 0, 0, 0, 0, fmt.Errorf("math.groupnorm: expected NCHW rank 4, got %d", len(shape))
	}

	batch, channels, height, width := shape[0], shape[1], shape[2], shape[3]

	if batch <= 0 || channels <= 0 || height <= 0 || width <= 0 {
		return 0, 0, 0, 0, fmt.Errorf("math.groupnorm: all NCHW dimensions must be positive")
	}

	if stateDict.Groups <= 0 {
		return 0, 0, 0, 0, fmt.Errorf("math.groupnorm: groups must be positive")
	}

	if channels%stateDict.Groups != 0 {
		return 0, 0, 0, 0, fmt.Errorf(
			"math.groupnorm: channels %d must be divisible by groups %d",
			channels,
			stateDict.Groups,
		)
	}

	inputLen := batch * channels * height * width

	if len(stateDict.Inputs[0]) != inputLen {
		return 0, 0, 0, 0, fmt.Errorf(
			"math.groupnorm: input length %d does not match NCHW size %d",
			len(stateDict.Inputs[0]),
			inputLen,
		)
	}

	if len(stateDict.Weight) != channels {
		return 0, 0, 0, 0, fmt.Errorf(
			"math.groupnorm: weight length %d does not match channels %d",
			len(stateDict.Weight),
			channels,
		)
	}

	if len(stateDict.Bias) != channels {
		return 0, 0, 0, 0, fmt.Errorf(
			"math.groupnorm: bias length %d does not match channels %d",
			len(stateDict.Bias),
			channels,
		)
	}

	return batch, channels, height, width, nil
}

func groupNormKernel(
	output,
	input,
	weight,
	bias []float64,
	eps float64,
	groups,
	batch,
	channels,
	height,
	width int,
) {
	channelsPerGroup := channels / groups
	spatial := height * width
	groupSize := channelsPerGroup * spatial

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		batchOffset := batchIndex * channels * spatial

		for groupIndex := 0; groupIndex < groups; groupIndex++ {
			channelStart := groupIndex * channelsPerGroup
			groupOffset := batchOffset + channelStart*spatial
			groupInput := input[groupOffset : groupOffset+groupSize]
			mean := reduceSum(groupInput) / float64(groupSize)
			variance := reduceSumSq(groupInput)/float64(groupSize) - mean*mean

			if variance < 0 {
				variance = 0
			}

			invStd := 1 / gomath.Sqrt(variance+eps)

			for channelOffset := 0; channelOffset < channelsPerGroup; channelOffset++ {
				channel := channelStart + channelOffset
				start := batchOffset + channel*spatial
				end := start + spatial
				scale := weight[channel] * invStd
				shift := bias[channel] - mean*scale

				groupNormAffine(output[start:end], input[start:end], scale, shift)
			}
		}
	}
}

func groupNormAffine(output, input []float64, scale, shift float64) {
	for index, value := range input {
		output[index] = value*scale + shift
	}
}
