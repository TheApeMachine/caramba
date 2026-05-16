//go:build !amd64 && !arm64

package shape

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
	upsampleNearest2DRowGenericKernel(dst, src, scaleW)
}
