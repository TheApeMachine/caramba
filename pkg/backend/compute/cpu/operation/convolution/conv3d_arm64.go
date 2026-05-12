//go:build arm64

package convolution

//go:noescape
func conv3dNEON(
	out, x, weight, bias []float64,
	n, inC, d, h, w, outC, kD, kH, kW, sD, sH, sW, pD, pH, pW, dilD, dilH, dilW, groups, dOut, hOut, wOut int,
)

func conv3dForward(
	x []float64, n, inC, d, h, w int,
	weight, bias []float64,
	outC, kD, kH, kW, sD, sH, sW, pD, pH, pW, dilD, dilH, dilW, groups int,
) []float64 {
	dOut := (d+2*pD-dilD*(kD-1)-1)/sD + 1
	hOut := (h+2*pH-dilH*(kH-1)-1)/sH + 1
	wOut := (w+2*pW-dilW*(kW-1)-1)/sW + 1
	out := make([]float64, n*outC*dOut*hOut*wOut)
	conv3dNEON(out, x, weight, bias, n, inC, d, h, w, outC, kD, kH, kW, sD, sH, sW, pD, pH, pW, dilD, dilH, dilW, groups, dOut, hOut, wOut)

	return out
}
