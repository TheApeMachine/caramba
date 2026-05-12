//go:build amd64

package convolution

//go:noescape
func conv2dAVX2(
	out, x, weight, bias []float64,
	n, inC, h, w, outC, kH, kW, strideH, strideW, padH, padW, dilH, dilW, groups, hOut, wOut int,
)

//go:noescape
func conv2dSSE2(
	out, x, weight, bias []float64,
	n, inC, h, w, outC, kH, kW, strideH, strideW, padH, padW, dilH, dilW, groups, hOut, wOut int,
)

func conv2dForward(
	x []float64, n, inC, h, w int,
	weight []float64, bias []float64,
	outC, kH, kW, strideH, strideW, padH, padW, dilationH, dilationW, groups int,
) []float64 {
	hOut := (h+2*padH-dilationH*(kH-1)-1)/strideH + 1
	wOut := (w+2*padW-dilationW*(kW-1)-1)/strideW + 1
	out := make([]float64, n*outC*hOut*wOut)

	if useAVX2 && useFMA {
		conv2dAVX2(out, x, weight, bias, n, inC, h, w, outC, kH, kW, strideH, strideW, padH, padW, dilationH, dilationW, groups, hOut, wOut)
	} else {
		conv2dSSE2(out, x, weight, bias, n, inC, h, w, outC, kH, kW, strideH, strideW, padH, padW, dilationH, dilationW, groups, hOut, wOut)
	}

	return out
}
