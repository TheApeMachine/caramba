//go:build amd64

package convolution

//go:noescape
func conv1dAVX2(
	out, x, weight, bias []float64,
	n, inC, l, outC, kSize, stride, pad, dilation, groups, lOut int,
)

//go:noescape
func conv1dSSE2(
	out, x, weight, bias []float64,
	n, inC, l, outC, kSize, stride, pad, dilation, groups, lOut int,
)

func conv1dForward(
	x []float64, n, inC, l int,
	weight, bias []float64,
	outC, kernelSize, stride, padding, dilation, groups int,
) []float64 {
	lOut := (l+2*padding-dilation*(kernelSize-1)-1)/stride + 1
	out := make([]float64, n*outC*lOut)

	if useAVX2 && useFMA {
		conv1dAVX2(out, x, weight, bias, n, inC, l, outC, kernelSize, stride, padding, dilation, groups, lOut)
	} else {
		conv1dSSE2(out, x, weight, bias, n, inC, l, outC, kernelSize, stride, padding, dilation, groups, lOut)
	}

	return out
}
