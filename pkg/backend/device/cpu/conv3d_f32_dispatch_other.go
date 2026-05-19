//go:build !arm64

package cpu

func Conv3DFloat32Native(
	config Conv3DConfig,
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
) {
	conv3DFloat32Scalar(
		config,
		inputView, weightView, biasView, outputView,
		batch, inChannels, inD, inH, inW,
		outChannels, kD, kH, kW, outD, outH, outW,
	)
}
