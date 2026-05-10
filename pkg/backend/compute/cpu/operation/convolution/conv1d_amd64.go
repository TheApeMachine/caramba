//go:build amd64

package convolution

// conv1dInnerAMD64 computes one output element of Conv1d using SIMD dot
// product when the input slice is contiguous.
//
// For Conv1d the kernel values stride through input by Dilation, so we cannot
// always call dotProduct directly; instead the Go loop in conv1d.go handles the
// dilation/padding boundary checks and uses the scalar path.  This file
// registers the SIMD helpers so the package can use them for the contiguous
// Dilation==1, Padding==0 fast path (called from conv1dForwardFast).
func conv1dForwardFast(
	x []float64, n, inC, l int,
	w []float64, bias []float64,
	outC, k, stride, groups int,
) []float64 {
	lOut := (l-k)/stride + 1
	icPerGroup := inC / groups
	ocPerGroup := outC / groups
	out := make([]float64, n*outC*lOut)

	for ni := 0; ni < n; ni++ {
		for g := 0; g < groups; g++ {
			ocStart := g * ocPerGroup
			icStart := g * icPerGroup
			for oc := ocStart; oc < ocStart+ocPerGroup; oc++ {
				wRow := w[oc*icPerGroup*k : (oc+1)*icPerGroup*k]
				b := bias[oc]
				for lo := 0; lo < lOut; lo++ {
					sum := b
					for ic := 0; ic < icPerGroup; ic++ {
						absIC := icStart + ic
						xSeg := x[ni*inC*l+absIC*l+lo*stride : ni*inC*l+absIC*l+lo*stride+k]
						wSeg := wRow[ic*k : (ic+1)*k]
						sum += dotProduct(xSeg, wSeg)
					}
					out[ni*outC*lOut+oc*lOut+lo] = sum
				}
			}
		}
	}
	return out
}
