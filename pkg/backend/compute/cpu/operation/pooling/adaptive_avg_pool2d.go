package pooling

// AdaptiveAvgPool2d pools a 4-D input [N, C, H, W] to a fixed output size [OutH, OutW].
// Each output cell averages a variable-sized region of the input.
type AdaptiveAvgPool2d struct {
	OutH, OutW int
}

// NewAdaptiveAvgPool2d creates an AdaptiveAvgPool2d.
func NewAdaptiveAvgPool2d(outH, outW int) *AdaptiveAvgPool2d {
	return &AdaptiveAvgPool2d{OutH: outH, OutW: outW}
}

// Forward computes AdaptiveAvgPool2d.
// shape = [N, C, H, W]; data[0] = flat input.
// Returns flat output of length N*C*OutH*OutW.
func (p *AdaptiveAvgPool2d) Forward(shape []int, data ...[]float64) []float64 {
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
					cnt := (hEnd - hStart) * (wEnd - wStart)
					var sum float64
					for ih := hStart; ih < hEnd; ih++ {
						for iw := wStart; iw < wEnd; iw++ {
							sum += x[baseIn+ih*W+iw]
						}
					}
					if cnt > 0 {
						out[baseOut+oh*Wout+ow] = sum / float64(cnt)
					}
				}
			}
		}
	}
	return out
}

// ceilDiv returns ceil((num * total) / denom) — used to compute adaptive region ends.
func ceilDiv(num, denom, total int) int {
	return (num*total + denom - 1) / denom
}
