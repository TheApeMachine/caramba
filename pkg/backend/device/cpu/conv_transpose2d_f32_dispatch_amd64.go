//go:build amd64

package cpu

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels/avx2"
)

func ConvTranspose2DFloat32Native(
	config Conv2DConfig,
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	avx2.ConvTranspose2DFloat32(
		avx2.Conv2DConfig{
			StrideH:   config.StrideH,
			StrideW:   config.StrideW,
			PaddingH:  config.PaddingH,
			PaddingW:  config.PaddingW,
			DilationH: config.DilationH,
			DilationW: config.DilationW,
		},
		inputView, weightView, biasView, outputView,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
	)
}
