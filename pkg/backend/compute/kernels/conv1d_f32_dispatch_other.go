//go:build !arm64

package kernels

func conv1DFloat32Native(
	config Conv1DConfig,
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inLength,
	outChannels, kernelLength, outLength int,
) {
	conv1DFloat32Scalar(
		config,
		inputView, weightView, biasView, outputView,
		batch, inChannels, inLength, outChannels, kernelLength, outLength,
	)
}
