package pooling

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

// AdaptiveMaxPool2d pools a 4-D input [N, C, H, W] to a fixed output size [OutH, OutW].
// Each output cell takes the maximum over a variable-sized region of the input.
type AdaptiveMaxPool2d struct {
}

// NewAdaptiveMaxPool2d creates an AdaptiveMaxPool2d.
func NewAdaptiveMaxPool2d(outH, outW int) *AdaptiveMaxPool2d {
	return &AdaptiveMaxPool2d{}
}

// Forward computes AdaptiveMaxPool2d.
// shape = [N, C, H, W]; data[0] = flat input.
// Returns flat output of length N*C*OutH*OutW.
func (pool *AdaptiveMaxPool2d) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("pooling.adaptive_max_pool2d"); err != nil {
		return nil, err
	}

	N, C, H, W, err := poolingShape4("pooling.adaptive_max_pool2d", stateDict)

	if err != nil {
		return nil, err
	}

	Hout, Wout := stateDict.OutH, stateDict.OutW

	if Hout <= 0 || Wout <= 0 {
		return nil, fmt.Errorf("pooling.adaptive_max_pool2d: output dimensions must be positive")
	}

	x := stateDict.Inputs[0]

	if len(x) != N*C*H*W {
		return nil, fmt.Errorf("pooling.adaptive_max_pool2d: input length does not match shape")
	}

	stateDict.EnsureOperationOutLen(N * C * Hout * Wout)

	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			baseIn := (n*C + c) * H * W
			baseOut := (n*C + c) * Hout * Wout
			for oh := 0; oh < Hout; oh++ {
				hStart := oh * H / Hout
				hEnd := ceilDiv(oh+1, Hout, H)
				for ow := 0; ow < Wout; ow++ {
					wStart := ow * W / Wout
					wEnd := ceilDiv(ow+1, Wout, W)
					maxVal := math.Inf(-1)
					for ih := hStart; ih < hEnd; ih++ {
						for iw := wStart; iw < wEnd; iw++ {
							v := x[baseIn+ih*W+iw]
							if v > maxVal {
								maxVal = v
							}
						}
					}
					stateDict.Out[baseOut+oh*Wout+ow] = maxVal
				}
			}
		}
	}

	return stateDict, nil
}
