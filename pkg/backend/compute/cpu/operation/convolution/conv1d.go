package convolution

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

// Conv1d applies a 1-D convolution over an input of shape [N, InC, L].
//
// Weight layout: [OutC, InC/Groups, K] (row-major, K is the innermost dim).
// Bias layout:   [OutC].
type Conv1d struct {
	Weight      []float64
	Bias        []float64
	InChannels  int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     int
	Dilation    int
	Groups      int
}

// NewConv1d allocates a Conv1d with Kaiming-uniform weight initialisation.
func NewConv1d(inC, outC, kernelSize, stride, padding, dilation, groups int) *Conv1d {
	if stride == 0 {
		stride = 1
	}
	if dilation == 0 {
		dilation = 1
	}
	if groups == 0 {
		groups = 1
	}
	fanIn := (inC / groups) * kernelSize
	wBound := math.Sqrt(2.0 / float64(fanIn))
	bBound := 1.0 / math.Sqrt(float64(fanIn))

	wSize := outC * (inC / groups) * kernelSize
	w := make([]float64, wSize)
	for i := range w {
		w[i] = (rand.Float64()*2 - 1) * wBound
	}
	b := make([]float64, outC)
	for i := range b {
		b[i] = (rand.Float64()*2 - 1) * bBound
	}
	return &Conv1d{
		Weight:      w,
		Bias:        b,
		InChannels:  inC,
		OutChannels: outC,
		KernelSize:  kernelSize,
		Stride:      stride,
		Padding:     padding,
		Dilation:    dilation,
		Groups:      groups,
	}
}

// Forward computes the 1-D convolution.
// shape = [N, InC, L]; data[0] = input flattened in that order.
// Returns output flattened as [N, OutC, L_out].
func (conv *Conv1d) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape := stateDict.OperationShape()

	if len(shape) < 3 {
		return nil, fmt.Errorf("convolution.conv1d: len(shape)=%d, need >= 3", len(shape))
	}

	if err := stateDict.RequireOperation("convolution.conv1d"); err != nil {
		return nil, err
	}

	batch := shape[0]
	inChannels := shape[1]
	length := shape[2]
	outChannels := stateDict.OutChannels
	kernelSize := stateDict.KernelSize
	stride := positiveDefault(stateDict.Stride, 1)
	padding := stateDict.Padding
	dilation := positiveDefault(stateDict.Dilation, 1)
	groups := positiveDefault(stateDict.Groups, 1)

	if stateDict.InChannels != 0 && stateDict.InChannels != inChannels {
		return nil, fmt.Errorf(
			"convolution.conv1d: shape in_channels=%d does not match state InChannels=%d",
			inChannels, stateDict.InChannels,
		)
	}

	if err := validateConv1dState(
		stateDict, batch, inChannels, length,
		outChannels, kernelSize, stride, padding, dilation, groups,
	); err != nil {
		return nil, err
	}

	output := conv1dForward(
		stateDict.Inputs[0], batch, inChannels, length,
		stateDict.Weight, stateDict.Bias,
		outChannels, kernelSize, stride, padding, dilation, groups,
	)

	stateDict.SetOperationOutput(output)

	return stateDict, nil
}

func validateConv1dState(
	stateDict *state.Dict,
	batch, inChannels, length int,
	outChannels, kernelSize, stride, padding, dilation, groups int,
) error {
	if batch <= 0 || inChannels <= 0 || length <= 0 || outChannels <= 0 ||
		kernelSize <= 0 || stride <= 0 || dilation <= 0 || groups <= 0 {
		return fmt.Errorf(
			"convolution.conv1d: invalid dimensions N=%d InC=%d L=%d OutC=%d K=%d stride=%d dilation=%d groups=%d",
			batch, inChannels, length, outChannels, kernelSize, stride, dilation, groups,
		)
	}

	if inChannels%groups != 0 || outChannels%groups != 0 {
		return fmt.Errorf("convolution.conv1d: groups=%d must divide InC=%d and OutC=%d", groups, inChannels, outChannels)
	}

	inputLength := batch * inChannels * length

	if len(stateDict.Inputs[0]) != inputLength {
		return fmt.Errorf("convolution.conv1d: len(input)=%d, need %d", len(stateDict.Inputs[0]), inputLength)
	}

	weightLength := outChannels * (inChannels / groups) * kernelSize

	if len(stateDict.Weight) != weightLength {
		return fmt.Errorf("convolution.conv1d: len(weight)=%d, need %d", len(stateDict.Weight), weightLength)
	}

	if len(stateDict.Bias) != outChannels {
		return fmt.Errorf("convolution.conv1d: len(bias)=%d, need OutC=%d", len(stateDict.Bias), outChannels)
	}

	lengthOut := (length+2*padding-dilation*(kernelSize-1)-1)/stride + 1

	if lengthOut <= 0 {
		return fmt.Errorf("convolution.conv1d: output length=%d must be positive", lengthOut)
	}

	return nil
}

func positiveDefault(value, defaultValue int) int {
	if value != 0 {
		return value
	}

	return defaultValue
}
