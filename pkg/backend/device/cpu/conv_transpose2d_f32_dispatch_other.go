//go:build !arm64 && !amd64

package cpu

func ConvTranspose2DFloat32Native(
	config Conv2DConfig,
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	convTranspose2DFloat32Scalar(
		config,
		inputView, weightView, biasView, outputView,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
	)
}
