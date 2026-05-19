//go:build !arm64

package cpu

func Pool2DFloat32Native(
	config PoolConfig,
	inputView, outputView []float32,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	pool2DFloat32Scalar(
		config,
		inputView, outputView,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		useMax,
	)
}
