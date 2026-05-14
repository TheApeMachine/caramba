package pooling

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

// MaxPool2d applies 2-D max pooling over a 4-D input [N, C, H, W].
type MaxPool2d struct{}

// NewMaxPool2d creates a stateless MaxPool2d operation.
func NewMaxPool2d(kernelH, kernelW, strideH, strideW, padH, padW, dilH, dilW int, ceil bool) *MaxPool2d {
	return &MaxPool2d{}
}

func outSizeMax(in, kernel, stride, pad, dilation int, ceil bool) int {
	eff := dilation*(kernel-1) + 1
	if ceil {
		return int(math.Ceil(float64(in+2*pad-eff)/float64(stride))) + 1
	}
	return (in+2*pad-eff)/stride + 1
}

func (pool *MaxPool2d) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("pooling.max_pool2d"); err != nil {
		return nil, err
	}

	N, C, H, W, err := poolingShape4("pooling.max_pool2d", stateDict)

	if err != nil {
		return nil, err
	}

	config, err := poolingConfig("pooling.max_pool2d", stateDict)

	if err != nil {
		return nil, err
	}

	Hout := outSizeMax(H, config.kernelH, config.strideH, config.padH, config.dilationH, stateDict.Ceil)
	Wout := outSizeMax(W, config.kernelW, config.strideW, config.padW, config.dilationW, stateDict.Ceil)

	if Hout <= 0 || Wout <= 0 {
		return nil, fmt.Errorf("pooling.max_pool2d: output spatial dimensions must be positive")
	}

	x := stateDict.Inputs[0]

	if len(x) != N*C*H*W {
		return nil, fmt.Errorf("pooling.max_pool2d: input length does not match shape")
	}

	stateDict.EnsureOperationOutLen(N * C * Hout * Wout)

	for batchIndex := 0; batchIndex < N; batchIndex++ {
		for channelIndex := 0; channelIndex < C; channelIndex++ {
			baseIn := (batchIndex*C + channelIndex) * H * W
			baseOut := (batchIndex*C + channelIndex) * Hout * Wout

			for outputH := 0; outputH < Hout; outputH++ {
				for outputW := 0; outputW < Wout; outputW++ {
					hStart := outputH*config.strideH - config.padH
					wStart := outputW*config.strideW - config.padW
					var rowBuf [maxKernelElems]float64
					count := 0

					for kernelH := 0; kernelH < config.kernelH; kernelH++ {
						inputH := hStart + kernelH*config.dilationH

						if inputH < 0 || inputH >= H {
							continue
						}

						rowStart := baseIn + inputH*W

						for kernelW := 0; kernelW < config.kernelW; kernelW++ {
							inputW := wStart + kernelW*config.dilationW

							if inputW < 0 || inputW >= W {
								continue
							}

							rowBuf[count] = x[rowStart+inputW]
							count++
						}
					}

					maxValue := math.Inf(-1)

					if count > 0 {
						maxValue = kernelMax(rowBuf[:count])
					}

					stateDict.Out[baseOut+outputH*Wout+outputW] = maxValue
				}
			}
		}
	}

	return stateDict, nil
}

const maxKernelElems = 1024
