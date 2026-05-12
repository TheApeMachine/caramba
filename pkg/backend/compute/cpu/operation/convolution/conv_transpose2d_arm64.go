//go:build arm64

package convolution

//go:noescape
func convTranspose2dNEON(
	out, x, wt, bias []float64,
	n, inC, h, w, outC, kH, kW, sH, sW, groups int,
)

func convTranspose2dForwardFast(
	x []float64, n, inC, h, w int,
	wt, bias []float64,
	outC, kH, kW, sH, sW, groups int,
) []float64 {
	hOut := (h-1)*sH + kH
	wOut := (w-1)*sW + kW
	out := make([]float64, n*outC*hOut*wOut)
	convTranspose2dNEON(out, x, wt, bias, n, inC, h, w, outC, kH, kW, sH, sW, groups)

	return out
}
