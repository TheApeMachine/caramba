package pooling

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

// AdaptiveAvgPool2d pools a 4-D input [N, C, H, W] to a fixed output size [OutH, OutW].
// Each output cell averages a variable-sized region of the input.
type AdaptiveAvgPool2d struct {
}

// NewAdaptiveAvgPool2d creates an AdaptiveAvgPool2d.
func NewAdaptiveAvgPool2d(outH, outW int) *AdaptiveAvgPool2d {
	return &AdaptiveAvgPool2d{}
}

// Forward computes AdaptiveAvgPool2d.
// shape = [N, C, H, W]; data[0] = flat input.
// Returns flat output of length N*C*OutH*OutW.
func (pool *AdaptiveAvgPool2d) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("pooling.adaptive_avg_pool2d"); err != nil {
		return nil, err
	}

	N, C, H, W, err := poolingShape4("pooling.adaptive_avg_pool2d", stateDict)

	if err != nil {
		return nil, err
	}

	Hout, Wout := stateDict.OutH, stateDict.OutW

	if Hout <= 0 || Wout <= 0 {
		return nil, fmt.Errorf("pooling.adaptive_avg_pool2d: output dimensions must be positive")
	}

	x := stateDict.Inputs[0]

	if len(x) != N*C*H*W {
		return nil, fmt.Errorf("pooling.adaptive_avg_pool2d: input length does not match shape")
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
					cnt := (hEnd - hStart) * (wEnd - wStart)
					var sum float64
					for ih := hStart; ih < hEnd; ih++ {
						for iw := wStart; iw < wEnd; iw++ {
							sum += x[baseIn+ih*W+iw]
						}
					}
					if cnt > 0 {
						stateDict.Out[baseOut+oh*Wout+ow] = sum / float64(cnt)
					}
				}
			}
		}
	}

	return stateDict, nil
}

// ceilDiv returns ceil((num * total) / denom) — used to compute adaptive region ends.
func ceilDiv(num, denom, total int) int {
	return (num*total + denom - 1) / denom
}
