//go:build amd64

package convolution

// conv3dForwardFast is the amd64 AVX2/SSE2-accelerated fast path for
// Dilation==1, Padding==0 Conv3d. Each output element's innermost
// kernel-width dot product is dispatched through dotProduct.
func conv3dForwardFast(
	x []float64, n, inC, d, h, w int,
	wt []float64, bias []float64,
	outC, kD, kH, kW, sD, sH, sW, groups int,
) []float64 {
	dOut := (d-kD)/sD + 1
	hOut := (h-kH)/sH + 1
	wOut := (w-kW)/sW + 1
	icPerGroup := inC / groups
	ocPerGroup := outC / groups
	out := make([]float64, n*outC*dOut*hOut*wOut)

	for ni := 0; ni < n; ni++ {
		for g := 0; g < groups; g++ {
			ocStart := g * ocPerGroup
			icStart := g * icPerGroup
			for oc := ocStart; oc < ocStart+ocPerGroup; oc++ {
				kernElems := icPerGroup * kD * kH * kW
				wRow := wt[oc*kernElems : (oc+1)*kernElems]
				b := bias[oc]
				for do := 0; do < dOut; do++ {
					for ho := 0; ho < hOut; ho++ {
						for wo := 0; wo < wOut; wo++ {
							sum := b
							for ic := 0; ic < icPerGroup; ic++ {
								absIC := icStart + ic
								for kd := 0; kd < kD; kd++ {
									di := do*sD + kd
									for kh := 0; kh < kH; kh++ {
										hi := ho*sH + kh
										xBase := ni*inC*d*h*w + absIC*d*h*w + di*h*w + hi*w + wo*sW
										wBase := ic*kD*kH*kW + kd*kH*kW + kh*kW
										sum += dotProduct(
											x[xBase:xBase+kW],
											wRow[wBase:wBase+kW],
										)
									}
								}
							}
							out[ni*outC*dOut*hOut*wOut+oc*dOut*hOut*wOut+do*hOut*wOut+ho*wOut+wo] = sum
						}
					}
				}
			}
		}
	}
	return out
}
