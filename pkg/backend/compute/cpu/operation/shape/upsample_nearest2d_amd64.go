//go:build amd64

package shape

//go:noescape
func upsampleNearest2DRowScale2AVX2(dst, src []float64)

//go:noescape
func upsampleNearest2DRowScale2SSE2(dst, src []float64)

func upsampleNearest2DKernel(
	dst []float64,
	src []float64,
	batch int,
	channels int,
	height int,
	width int,
	scaleH int,
	scaleW int,
) {
	upsampleNearest2DGenericKernel(
		dst, src, batch, channels, height, width, scaleH, scaleW,
	)
}

func upsampleNearest2DRowKernel(dst []float64, src []float64, scaleW int) {
	if scaleW != 2 {
		upsampleNearest2DRowGenericKernel(dst, src, scaleW)

		return
	}

	if useAVX2 {
		upsampleNearest2DRowScale2AVX2(dst, src)

		return
	}

	upsampleNearest2DRowScale2SSE2(dst, src)
}
