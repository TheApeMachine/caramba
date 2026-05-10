//go:build amd64

package convolution

// conv2dForwardFast is the amd64 AVX2/SSE2-accelerated fast path for
// Dilation==1, Padding==0 Conv2d.  Each output element's innermost
// kernel-width dot product is dispatched through dotProduct.
func conv2dForwardFast(
	x []float64, n, inC, h, w int,
	wt []float64, bias []float64,
	outC, kH, kW, strideH, strideW, groups int,
) []float64 {
	hOut := (h-kH)/strideH + 1
	wOut := (w-kW)/strideW + 1
	icPerGroup := inC / groups
	ocPerGroup := outC / groups
	out := make([]float64, n*outC*hOut*wOut)

	for ni := 0; ni < n; ni++ {
		for g := 0; g < groups; g++ {
			ocStart := g * ocPerGroup
			icStart := g * icPerGroup
			for oc := ocStart; oc < ocStart+ocPerGroup; oc++ {
				kernElems := icPerGroup * kH * kW
				wRow := wt[oc*kernElems : (oc+1)*kernElems]
				b := bias[oc]
				for ho := 0; ho < hOut; ho++ {
					for wo := 0; wo < wOut; wo++ {
						sum := b
						for ic := 0; ic < icPerGroup; ic++ {
							absIC := icStart + ic
							for kh := 0; kh < kH; kh++ {
								hi := ho*strideH + kh
								xBase := ni*inC*h*w + absIC*h*w + hi*w + wo*strideW
								wBase := ic*kH*kW + kh*kW
								sum += dotProduct(
									x[xBase:xBase+kW],
									wRow[wBase:wBase+kW],
								)
							}
						}
						out[ni*outC*hOut*wOut+oc*hOut*wOut+ho*wOut+wo] = sum
					}
				}
			}
		}
	}
	return out
}
