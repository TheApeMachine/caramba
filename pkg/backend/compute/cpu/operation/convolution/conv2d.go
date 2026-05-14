package convolution

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

// Conv2d applies a 2-D convolution over input [N, InC, H, W].
//
// Weight layout: [OutC, InC/Groups, KH, KW] (row-major).
// Bias layout:   [OutC].
type Conv2d struct {
	Weight      []float64
	Bias        []float64
	InChannels  int
	OutChannels int
	KernelH     int
	KernelW     int
	StrideH     int
	StrideW     int
	PadH        int
	PadW        int
	DilationH   int
	DilationW   int
	Groups      int
}

// NewConv2d allocates a Conv2d with Kaiming-uniform weight initialisation.
func NewConv2d(inC, outC, kH, kW, strideH, strideW, padH, padW, dilH, dilW, groups int) *Conv2d {
	if strideH == 0 {
		strideH = 1
	}
	if strideW == 0 {
		strideW = 1
	}
	if dilH == 0 {
		dilH = 1
	}
	if dilW == 0 {
		dilW = 1
	}
	if groups == 0 {
		groups = 1
	}
	fanIn := (inC / groups) * kH * kW
	wBound := math.Sqrt(2.0 / float64(fanIn))
	bBound := 1.0 / math.Sqrt(float64(fanIn))

	wSize := outC * (inC / groups) * kH * kW
	w := make([]float64, wSize)
	for i := range w {
		w[i] = (rand.Float64()*2 - 1) * wBound
	}
	b := make([]float64, outC)
	for i := range b {
		b[i] = (rand.Float64()*2 - 1) * bBound
	}
	return &Conv2d{
		Weight:      w,
		Bias:        b,
		InChannels:  inC,
		OutChannels: outC,
		KernelH:     kH,
		KernelW:     kW,
		StrideH:     strideH,
		StrideW:     strideW,
		PadH:        padH,
		PadW:        padW,
		DilationH:   dilH,
		DilationW:   dilW,
		Groups:      groups,
	}
}

// Forward computes the 2-D convolution.
// shape = [N, InC, H, W]; data[0] = input.
// Returns [N, OutC, H_out, W_out].
func (conv *Conv2d) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape := stateDict.OperationShape()

	if len(shape) < 4 {
		return nil, fmt.Errorf("convolution.conv2d: len(shape)=%d, need >= 4", len(shape))
	}

	if err := stateDict.RequireOperation("convolution.conv2d"); err != nil {
		return nil, err
	}

	batch := shape[0]
	inChannels := shape[1]
	height := shape[2]
	width := shape[3]
	outChannels := stateDict.OutChannels
	kernelH := stateDict.KernelH
	kernelW := stateDict.KernelW
	strideH := positiveDefault(stateDict.StrideH, 1)
	strideW := positiveDefault(stateDict.StrideW, 1)
	dilationH := positiveDefault(stateDict.DilationH, 1)
	dilationW := positiveDefault(stateDict.DilationW, 1)
	groups := positiveDefault(stateDict.Groups, 1)

	if stateDict.InChannels != 0 && stateDict.InChannels != inChannels {
		return nil, fmt.Errorf(
			"convolution.conv2d: shape in_channels=%d does not match state InChannels=%d",
			inChannels, stateDict.InChannels,
		)
	}

	if err := validateConv2dState(
		stateDict, batch, inChannels, height, width,
		outChannels, kernelH, kernelW,
		strideH, strideW, stateDict.PadH, stateDict.PadW,
		dilationH, dilationW, groups,
	); err != nil {
		return nil, err
	}

	output := conv2dForward(
		stateDict.Inputs[0], batch, inChannels, height, width,
		stateDict.Weight, stateDict.Bias,
		outChannels, kernelH, kernelW,
		strideH, strideW, stateDict.PadH, stateDict.PadW, dilationH, dilationW,
		groups,
	)

	stateDict.SetOperationOutput(output)

	return stateDict, nil
}

func validateConv2dState(
	stateDict *state.Dict,
	batch, inChannels, height, width int,
	outChannels, kernelH, kernelW int,
	strideH, strideW, padH, padW, dilationH, dilationW, groups int,
) error {
	if batch <= 0 || inChannels <= 0 || height <= 0 || width <= 0 ||
		outChannels <= 0 || kernelH <= 0 || kernelW <= 0 ||
		strideH <= 0 || strideW <= 0 || dilationH <= 0 || dilationW <= 0 ||
		groups <= 0 {
		return fmt.Errorf("convolution.conv2d: invalid dimensions")
	}

	if inChannels%groups != 0 || outChannels%groups != 0 {
		return fmt.Errorf("convolution.conv2d: groups=%d must divide InC=%d and OutC=%d", groups, inChannels, outChannels)
	}

	inputLength := batch * inChannels * height * width

	if len(stateDict.Inputs[0]) != inputLength {
		return fmt.Errorf("convolution.conv2d: len(input)=%d, need %d", len(stateDict.Inputs[0]), inputLength)
	}

	weightLength := outChannels * (inChannels / groups) * kernelH * kernelW

	if len(stateDict.Weight) != weightLength {
		return fmt.Errorf("convolution.conv2d: len(weight)=%d, need %d", len(stateDict.Weight), weightLength)
	}

	if len(stateDict.Bias) != outChannels {
		return fmt.Errorf("convolution.conv2d: len(bias)=%d, need OutC=%d", len(stateDict.Bias), outChannels)
	}

	heightOut := (height+2*padH-dilationH*(kernelH-1)-1)/strideH + 1
	widthOut := (width+2*padW-dilationW*(kernelW-1)-1)/strideW + 1

	if heightOut <= 0 || widthOut <= 0 {
		return fmt.Errorf("convolution.conv2d: output shape [%d,%d] must be positive", heightOut, widthOut)
	}

	return nil
}
