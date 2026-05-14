package convolution

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

// Conv3d applies a 3-D convolution over input [N, InC, D, H, W].
//
// Weight layout: [OutC, InC/Groups, KD, KH, KW] (row-major).
// Bias layout:   [OutC].
type Conv3d struct {
	Weight      []float64
	Bias        []float64
	InChannels  int
	OutChannels int
	KernelD     int
	KernelH     int
	KernelW     int
	StrideD     int
	StrideH     int
	StrideW     int
	PadD        int
	PadH        int
	PadW        int
	DilationD   int
	DilationH   int
	DilationW   int
	Groups      int
}

// NewConv3d allocates a Conv3d with Kaiming-uniform weight initialisation.
func NewConv3d(inC, outC, kD, kH, kW, sD, sH, sW, pD, pH, pW, dilD, dilH, dilW, groups int) *Conv3d {
	if sD == 0 {
		sD = 1
	}
	if sH == 0 {
		sH = 1
	}
	if sW == 0 {
		sW = 1
	}
	if dilD == 0 {
		dilD = 1
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
	fanIn := (inC / groups) * kD * kH * kW
	wBound := math.Sqrt(2.0 / float64(fanIn))
	bBound := 1.0 / math.Sqrt(float64(fanIn))

	wSize := outC * (inC / groups) * kD * kH * kW
	w := make([]float64, wSize)
	for i := range w {
		w[i] = (rand.Float64()*2 - 1) * wBound
	}
	b := make([]float64, outC)
	for i := range b {
		b[i] = (rand.Float64()*2 - 1) * bBound
	}
	return &Conv3d{
		Weight:      w,
		Bias:        b,
		InChannels:  inC,
		OutChannels: outC,
		KernelD:     kD,
		KernelH:     kH,
		KernelW:     kW,
		StrideD:     sD,
		StrideH:     sH,
		StrideW:     sW,
		PadD:        pD,
		PadH:        pH,
		PadW:        pW,
		DilationD:   dilD,
		DilationH:   dilH,
		DilationW:   dilW,
		Groups:      groups,
	}
}

// Forward computes the 3-D convolution.
// shape = [N, InC, D, H, W]; data[0] = input.
// Returns [N, OutC, D_out, H_out, W_out].
func (conv *Conv3d) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape := stateDict.OperationShape()

	if len(shape) < 5 {
		return nil, fmt.Errorf("convolution.conv3d: len(shape)=%d, need >= 5", len(shape))
	}

	if err := stateDict.RequireOperation("convolution.conv3d"); err != nil {
		return nil, err
	}

	batch := shape[0]
	inChannels := shape[1]
	depth := shape[2]
	height := shape[3]
	width := shape[4]
	outChannels := stateDict.OutChannels
	kernelD := stateDict.KernelD
	kernelH := stateDict.KernelH
	kernelW := stateDict.KernelW
	strideD := positiveDefault(stateDict.StrideD, 1)
	strideH := positiveDefault(stateDict.StrideH, 1)
	strideW := positiveDefault(stateDict.StrideW, 1)
	dilationD := positiveDefault(stateDict.DilationD, 1)
	dilationH := positiveDefault(stateDict.DilationH, 1)
	dilationW := positiveDefault(stateDict.DilationW, 1)
	groups := positiveDefault(stateDict.Groups, 1)

	if stateDict.InChannels != 0 && stateDict.InChannels != inChannels {
		return nil, fmt.Errorf(
			"convolution.conv3d: shape in_channels=%d does not match state InChannels=%d",
			inChannels, stateDict.InChannels,
		)
	}

	if err := validateConv3dState(
		stateDict, batch, inChannels, depth, height, width,
		outChannels, kernelD, kernelH, kernelW,
		strideD, strideH, strideW,
		stateDict.PadD, stateDict.PadH, stateDict.PadW,
		dilationD, dilationH, dilationW, groups,
	); err != nil {
		return nil, err
	}

	output := conv3dForward(
		stateDict.Inputs[0], batch, inChannels, depth, height, width,
		stateDict.Weight, stateDict.Bias,
		outChannels, kernelD, kernelH, kernelW,
		strideD, strideH, strideW,
		stateDict.PadD, stateDict.PadH, stateDict.PadW,
		dilationD, dilationH, dilationW,
		groups,
	)

	stateDict.SetOperationOutput(output)

	return stateDict, nil
}

func validateConv3dState(
	stateDict *state.Dict,
	batch, inChannels, depth, height, width int,
	outChannels, kernelD, kernelH, kernelW int,
	strideD, strideH, strideW, padD, padH, padW int,
	dilationD, dilationH, dilationW, groups int,
) error {
	if batch <= 0 || inChannels <= 0 || depth <= 0 || height <= 0 || width <= 0 ||
		outChannels <= 0 || kernelD <= 0 || kernelH <= 0 || kernelW <= 0 ||
		strideD <= 0 || strideH <= 0 || strideW <= 0 ||
		dilationD <= 0 || dilationH <= 0 || dilationW <= 0 || groups <= 0 {
		return fmt.Errorf("convolution.conv3d: invalid dimensions")
	}

	if inChannels%groups != 0 || outChannels%groups != 0 {
		return fmt.Errorf("convolution.conv3d: groups=%d must divide InC=%d and OutC=%d", groups, inChannels, outChannels)
	}

	inputLength := batch * inChannels * depth * height * width

	if len(stateDict.Inputs[0]) != inputLength {
		return fmt.Errorf("convolution.conv3d: len(input)=%d, need %d", len(stateDict.Inputs[0]), inputLength)
	}

	weightLength := outChannels * (inChannels / groups) * kernelD * kernelH * kernelW

	if len(stateDict.Weight) != weightLength {
		return fmt.Errorf("convolution.conv3d: len(weight)=%d, need %d", len(stateDict.Weight), weightLength)
	}

	if len(stateDict.Bias) != outChannels {
		return fmt.Errorf("convolution.conv3d: len(bias)=%d, need OutC=%d", len(stateDict.Bias), outChannels)
	}

	depthOut := (depth+2*padD-dilationD*(kernelD-1)-1)/strideD + 1
	heightOut := (height+2*padH-dilationH*(kernelH-1)-1)/strideH + 1
	widthOut := (width+2*padW-dilationW*(kernelW-1)-1)/strideW + 1

	if depthOut <= 0 || heightOut <= 0 || widthOut <= 0 {
		return fmt.Errorf("convolution.conv3d: output shape [%d,%d,%d] must be positive", depthOut, heightOut, widthOut)
	}

	return nil
}
