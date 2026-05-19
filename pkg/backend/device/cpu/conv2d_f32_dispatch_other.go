//go:build !arm64

package cpu

func Conv2DFloat32Native(
	config Conv2DConfig,
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	conv2DFloat32Scalar(
		config,
		inputView, weightView, biasView, outputView,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
	)
}
