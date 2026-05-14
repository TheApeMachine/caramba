//go:build amd64

package activation

//go:noescape
func SigmoidAVX2(dst, src []float64)

//go:noescape
func SigmoidSSE2(dst, src []float64)

func sigmoidKernel(dst, src []float64) {
	elementCount := len(src)

	if len(dst) < elementCount {
		elementCount = len(dst)
	}

	width := 2
	vectorSigmoid := SigmoidSSE2

	if useAVX2 {
		width = 4
		vectorSigmoid = SigmoidAVX2
	}

	limit := elementCount / width * width

	if limit > 0 {
		vectorSigmoid(dst[:limit], src[:limit])
	}

	scalarSigmoid(dst[limit:elementCount], src[limit:elementCount])
}
