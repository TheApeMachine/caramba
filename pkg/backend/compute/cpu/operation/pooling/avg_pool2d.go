package pooling

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

// AvgPool2d applies 2-D average pooling over a 4-D input [N, C, H, W].
type AvgPool2d struct{}

// NewAvgPool2d creates a stateless AvgPool2d operation.
func NewAvgPool2d(
	kernelH, kernelW, strideH, strideW, padH, padW, dilH, dilW int,
	ceil, countIncludePad bool, divisorOverride int,
) *AvgPool2d {
	return &AvgPool2d{}
}

func (pool *AvgPool2d) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("pooling.avg_pool2d"); err != nil {
		return nil, err
	}

	N, C, H, W, err := poolingShape4("pooling.avg_pool2d", stateDict)

	if err != nil {
		return nil, err
	}

	config, err := poolingConfig("pooling.avg_pool2d", stateDict)

	if err != nil {
		return nil, err
	}

	Hout := outSizeMax(H, config.kernelH, config.strideH, config.padH, config.dilationH, stateDict.Ceil)
	Wout := outSizeMax(W, config.kernelW, config.strideW, config.padW, config.dilationW, stateDict.Ceil)

	if Hout <= 0 || Wout <= 0 {
		return nil, fmt.Errorf("pooling.avg_pool2d: output spatial dimensions must be positive")
	}

	x := stateDict.Inputs[0]

	if len(x) != N*C*H*W {
		return nil, fmt.Errorf("pooling.avg_pool2d: input length does not match shape")
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
					kernelCount := 0

					for kernelH := 0; kernelH < config.kernelH; kernelH++ {
						inputH := hStart + kernelH*config.dilationH
						validH := inputH >= 0 && inputH < H

						for kernelW := 0; kernelW < config.kernelW; kernelW++ {
							kernelCount++
							inputW := wStart + kernelW*config.dilationW

							if validH && inputW >= 0 && inputW < W {
								rowBuf[count] = x[baseIn+inputH*W+inputW]
								count++
							}
						}
					}

					divisor := count

					if stateDict.Divisor != 0 {
						divisor = stateDict.Divisor
					} else if stateDict.CountPad {
						divisor = kernelCount
					}

					avg := math.NaN()

					if count > 0 && divisor > 0 {
						avg = kernelSum(rowBuf[:count]) / float64(divisor)
					} else if divisor > 0 {
						avg = 0
					}

					stateDict.Out[baseOut+outputH*Wout+outputW] = avg
				}
			}
		}
	}

	return stateDict, nil
}
