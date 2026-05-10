//go:build amd64

package convolution

// convTranspose2dForwardFast is the amd64 AVX2/SSE2-accelerated fast path for
// Dilation==1, Padding==0 ConvTranspose2d.
func convTranspose2dForwardFast(
	x []float64, n, inC, h, w int,
	wt []float64, bias []float64,
	outC, kH, kW, sH, sW, groups int,
) []float64 {
	hOut := (h-1)*sH + kH
	wOut := (w-1)*sW + kW
	ocPerGroup := outC / groups
	icPerGroup := inC / groups
	out := make([]float64, n*outC*hOut*wOut)

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

	for ni := 0; ni < n; ni++ {
		for g := 0; g < groups; g++ {
			icStart := g * icPerGroup
			ocStart := g * ocPerGroup
			for ic := 0; ic < icPerGroup; ic++ {
				absIC := icStart + ic
				kernElems := ocPerGroup * kH * kW
				wRow := wt[absIC*kernElems : (absIC+1)*kernElems]
				for hi := 0; hi < h; hi++ {
					for wi := 0; wi < w; wi++ {
						xVal := x[ni*inC*h*w+absIC*h*w+hi*w+wi]
						for oc := 0; oc < ocPerGroup; oc++ {
							absOC := ocStart + oc
							for kh := 0; kh < kH; kh++ {
								ho := hi*sH + kh
								oBase := ni*outC*hOut*wOut + absOC*hOut*wOut + ho*wOut + wi*sW
								wBase := oc*kH*kW + kh*kW
								scaledAdd(
									out[oBase:oBase+kW],
									wRow[wBase:wBase+kW],
									xVal,
								)
							}
						}
					}
				}
			}
		}
	}
	return out
}
