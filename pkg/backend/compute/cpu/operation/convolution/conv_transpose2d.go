package convolution

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

// ConvTranspose2d applies a 2-D transposed convolution over input [N, InC, H, W].
//
// Weight layout: [InC, OutC/Groups, KH, KW] (row-major).
// Bias layout:   [OutC].
type ConvTranspose2d struct {
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
	OutPadH     int
	OutPadW     int
	DilationH   int
	DilationW   int
	Groups      int
}

// NewConvTranspose2d allocates a ConvTranspose2d with Kaiming-uniform weight initialisation.
func NewConvTranspose2d(inC, outC, kH, kW, sH, sW, pH, pW, outPadH, outPadW, dilH, dilW, groups int) *ConvTranspose2d {
	if sH == 0 {
		sH = 1
	}
	if sW == 0 {
		sW = 1
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
	fanIn := (outC / groups) * kH * kW
	wBound := math.Sqrt(2.0 / float64(fanIn))
	bBound := 1.0 / math.Sqrt(float64(fanIn))

	wSize := inC * (outC / groups) * kH * kW
	w := make([]float64, wSize)
	for i := range w {
		w[i] = (rand.Float64()*2 - 1) * wBound
	}
	b := make([]float64, outC)
	for i := range b {
		b[i] = (rand.Float64()*2 - 1) * bBound
	}
	return &ConvTranspose2d{
		Weight:      w,
		Bias:        b,
		InChannels:  inC,
		OutChannels: outC,
		KernelH:     kH,
		KernelW:     kW,
		StrideH:     sH,
		StrideW:     sW,
		PadH:        pH,
		PadW:        pW,
		OutPadH:     outPadH,
		OutPadW:     outPadW,
		DilationH:   dilH,
		DilationW:   dilW,
		Groups:      groups,
	}
}

// Forward computes the 2-D transposed convolution.
// shape = [N, InC, H, W]; data[0] = input.
// Returns [N, OutC, H_out, W_out] where:
//
//	H_out = (H-1)*StrideH - 2*PadH + DilationH*(KH-1) + OutPadH + 1
//	W_out = (W-1)*StrideW - 2*PadW + DilationW*(KW-1) + OutPadW + 1
func (conv *ConvTranspose2d) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape := stateDict.OperationShape()

	if len(shape) < 4 {
		return nil, fmt.Errorf("convolution.conv_transpose2d: len(shape)=%d, need >= 4", len(shape))
	}

	if err := stateDict.RequireOperation("convolution.conv_transpose2d"); err != nil {
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

	if err := validateConvTranspose2dState(
		stateDict, batch, inChannels, height, width,
		outChannels, kernelH, kernelW,
		strideH, strideW, stateDict.PadH, stateDict.PadW,
		stateDict.OutPadH, stateDict.OutPadW,
		dilationH, dilationW, groups,
	); err != nil {
		return nil, err
	}

	input := stateDict.Inputs[0]

	if dilationH == 1 && dilationW == 1 &&
		stateDict.PadH == 0 && stateDict.PadW == 0 &&
		stateDict.OutPadH == 0 && stateDict.OutPadW == 0 {
		output := convTranspose2dForwardFast(
			input, batch, inChannels, height, width,
			stateDict.Weight, stateDict.Bias,
			outChannels, kernelH, kernelW,
			strideH, strideW, groups,
		)

		stateDict.SetOperationOutput(output)

		return stateDict, nil
	}

	heightOut := (height-1)*strideH - 2*stateDict.PadH + dilationH*(kernelH-1) + stateDict.OutPadH + 1
	widthOut := (width-1)*strideW - 2*stateDict.PadW + dilationW*(kernelW-1) + stateDict.OutPadW + 1

	output := make([]float64, batch*outChannels*heightOut*widthOut)
	applyConvTranspose2d(
		output, input, stateDict.Weight, stateDict.Bias,
		batch, inChannels, height, width,
		outChannels, kernelH, kernelW,
		strideH, strideW,
		stateDict.PadH, stateDict.PadW,
		dilationH, dilationW,
		groups,
		heightOut, widthOut,
	)
	stateDict.SetOperationOutput(output)

	return stateDict, nil
}

func validateConvTranspose2dState(
	stateDict *state.Dict,
	batch, inChannels, height, width int,
	outChannels, kernelH, kernelW int,
	strideH, strideW, padH, padW, outPadH, outPadW int,
	dilationH, dilationW, groups int,
) error {
	if batch <= 0 || inChannels <= 0 || height <= 0 || width <= 0 ||
		outChannels <= 0 || kernelH <= 0 || kernelW <= 0 ||
		strideH <= 0 || strideW <= 0 || dilationH <= 0 || dilationW <= 0 ||
		groups <= 0 {
		return fmt.Errorf("convolution.conv_transpose2d: invalid dimensions")
	}

	if inChannels%groups != 0 || outChannels%groups != 0 {
		return fmt.Errorf(
			"convolution.conv_transpose2d: groups=%d must divide InC=%d and OutC=%d",
			groups, inChannels, outChannels,
		)
	}

	inputLength := batch * inChannels * height * width

	if len(stateDict.Inputs[0]) != inputLength {
		return fmt.Errorf("convolution.conv_transpose2d: len(input)=%d, need %d", len(stateDict.Inputs[0]), inputLength)
	}

	weightLength := inChannels * (outChannels / groups) * kernelH * kernelW

	if len(stateDict.Weight) != weightLength {
		return fmt.Errorf(
			"convolution.conv_transpose2d: len(weight)=%d, need %d",
			len(stateDict.Weight), weightLength,
		)
	}

	if len(stateDict.Bias) != outChannels {
		return fmt.Errorf("convolution.conv_transpose2d: len(bias)=%d, need OutC=%d", len(stateDict.Bias), outChannels)
	}

	heightOut := (height-1)*strideH - 2*padH + dilationH*(kernelH-1) + outPadH + 1
	widthOut := (width-1)*strideW - 2*padW + dilationW*(kernelW-1) + outPadW + 1

	if heightOut <= 0 || widthOut <= 0 {
		return fmt.Errorf("convolution.conv_transpose2d: output shape [%d,%d] must be positive", heightOut, widthOut)
	}

	return nil
}

// applyConvTranspose2d is the pure-Go scatter-add implementation.
// For each (n, ic, h, w) input pixel, scatter-add to output positions.
func applyConvTranspose2d(
	out, x, wt, bias []float64,
	n, inC, h, w int,
	outC, kH, kW int,
	sH, sW int,
	pH, pW int,
	dilH, dilW int,
	groups int,
	hOut, wOut int,
) {
	ocPerGroup := outC / groups
	icPerGroup := inC / groups

	// Initialize bias.
	for ni := 0; ni < n; ni++ {
		for oc := 0; oc < outC; oc++ {
			b := bias[oc]
			base := ni*outC*hOut*wOut + oc*hOut*wOut
			for i := 0; i < hOut*wOut; i++ {
				out[base+i] = b
			}
		}
	}

	// Scatter-add from input to output.
	for ni := 0; ni < n; ni++ {
		for g := 0; g < groups; g++ {
			icStart := g * icPerGroup
			ocStart := g * ocPerGroup
			for ic := 0; ic < icPerGroup; ic++ {
				absIC := icStart + ic
				// weight row for this input channel: [ocPerGroup * kH * kW]
				kernElems := ocPerGroup * kH * kW
				wRow := wt[absIC*kernElems : (absIC+1)*kernElems]
				for hi := 0; hi < h; hi++ {
					for wi := 0; wi < w; wi++ {
						xVal := x[ni*inC*h*w+absIC*h*w+hi*w+wi]
						for oc := 0; oc < ocPerGroup; oc++ {
							absOC := ocStart + oc
							for kh := 0; kh < kH; kh++ {
								ho := hi*sH + kh*dilH - pH
								if ho < 0 || ho >= hOut {
									continue
								}
								for kw := 0; kw < kW; kw++ {
									wo := wi*sW + kw*dilW - pW
									if wo < 0 || wo >= wOut {
										continue
									}
									wIdx := oc*kH*kW + kh*kW + kw
									out[ni*outC*hOut*wOut+absOC*hOut*wOut+ho*wOut+wo] += xVal * wRow[wIdx]
								}
							}
						}
					}
				}
			}
		}
	}
}
