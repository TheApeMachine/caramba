//go:build arm64

package cpu

func Conv3DFloat32Native(
	config Conv3DConfig,
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
) {
	patchLength := inChannels * kD * kH * kW
	patchScratch := BorrowFloat32Buffer(patchLength)
	defer ReleaseFloat32Buffer(patchScratch)

	for batchIndex := range batch {
		inputBatchOffset := batchIndex * inChannels * inD * inH * inW

		for outChIndex := range outChannels {
			weightOffset := outChIndex * inChannels * kD * kH * kW

			for outDIndex := range outD {
				for outHIndex := range outH {
					for outWIndex := range outW {
						conv3DPatchGather(
							config,
							inputView, inputBatchOffset,
							patchScratch,
							inChannels, inD, inH, inW,
							kD, kH, kW,
							outDIndex, outHIndex, outWIndex,
						)

						dotValue := Conv3dPatchDotNEONAsm(
							&weightView[weightOffset],
							&patchScratch[0],
							patchLength,
						)

						outputView[(((batchIndex*outChannels+outChIndex)*outD+outDIndex)*outH+outHIndex)*outW+outWIndex] =
							biasView[outChIndex] + dotValue
					}
				}
			}
		}
	}
}
