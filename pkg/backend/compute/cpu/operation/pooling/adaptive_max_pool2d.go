package pooling

import "math"

// AdaptiveMaxPool2d pools a 4-D input [N, C, H, W] to a fixed output size [OutH, OutW].
// Each output cell takes the maximum over a variable-sized region of the input.
type AdaptiveMaxPool2d struct {
	OutH, OutW int
}

// NewAdaptiveMaxPool2d creates an AdaptiveMaxPool2d.
func NewAdaptiveMaxPool2d(outH, outW int) *AdaptiveMaxPool2d {
	return &AdaptiveMaxPool2d{OutH: outH, OutW: outW}
}

// Forward computes AdaptiveMaxPool2d.
// shape = [N, C, H, W]; data[0] = flat input.
// Returns flat output of length N*C*OutH*OutW.
func (p *AdaptiveMaxPool2d) Forward(shape []int, data ...[]float64) []float64 {
	N, C, H, W := shape[0], shape[1], shape[2], shape[3]
	Hout, Wout := p.OutH, p.OutW
	x := data[0]
	out := make([]float64, N*C*Hout*Wout)

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
					out[baseOut+oh*Wout+ow] = maxVal
				}
			}
		}
	}
	return out
}
