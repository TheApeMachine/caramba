//go:build darwin && cgo

package metal

import "fmt"

type metalConvolutionContract struct {
	operation           string
	inputLength         int
	weightLength        int
	biasLength          int
	expectedInputLength int
	expectedWeight      int
	expectedBias        int
	outputShape         []int
	expectedOutputShape []int
}

func (contract metalConvolutionContract) Validate() error {
	if err := contract.requireLength(
		"input", contract.inputLength, contract.expectedInputLength,
	); err != nil {
		return err
	}

	if err := contract.requireLength(
		"weight", contract.weightLength, contract.expectedWeight,
	); err != nil {
		return err
	}

	if err := contract.requireLength(
		"bias", contract.biasLength, contract.expectedBias,
	); err != nil {
		return err
	}

	if len(contract.outputShape) != len(contract.expectedOutputShape) {
		return fmt.Errorf(
			"%s: output rank %d does not match expected rank %d",
			contract.operation,
			len(contract.outputShape),
			len(contract.expectedOutputShape),
		)
	}

	for axis, actual := range contract.outputShape {
		expected := contract.expectedOutputShape[axis]

		if expected <= 0 {
			return fmt.Errorf(
				"%s: output shape %v must be positive",
				contract.operation,
				contract.expectedOutputShape,
			)
		}

		if actual == expected {
			continue
		}

		return fmt.Errorf(
			"%s: output shape %v does not match expected %v",
			contract.operation,
			contract.outputShape,
			contract.expectedOutputShape,
		)
	}

	return nil
}

func (contract metalConvolutionContract) requireLength(
	name string,
	actual int,
	expected int,
) error {
	if actual == expected {
		return nil
	}

	return fmt.Errorf("%s: len(%s)=%d, need %d", contract.operation, name, actual, expected)
}

func validateMetalConv1d(
	input []float64,
	weight []float64,
	bias []float64,
	batch int,
	inChannels int,
	length int,
	outChannels int,
	kernelSize int,
	stride int,
	padding int,
	dilation int,
	groups int,
	lengthOut int,
) error {
	const operation = "metal.conv1d"

	if batch <= 0 || inChannels <= 0 || length <= 0 || outChannels <= 0 ||
		kernelSize <= 0 || stride <= 0 || dilation <= 0 || groups <= 0 {
		return fmt.Errorf("%s: invalid dimensions", operation)
	}

	if err := validateMetalConvGroups(
		operation, inChannels, outChannels, groups,
	); err != nil {
		return err
	}

	expectedLengthOut := (length+2*padding-dilation*(kernelSize-1)-1)/stride + 1

	return metalConvolutionContract{
		operation:           operation,
		inputLength:         len(input),
		weightLength:        len(weight),
		biasLength:          len(bias),
		expectedInputLength: batch * inChannels * length,
		expectedWeight:      outChannels * (inChannels / groups) * kernelSize,
		expectedBias:        outChannels,
		outputShape:         []int{lengthOut},
		expectedOutputShape: []int{expectedLengthOut},
	}.Validate()
}

func validateMetalConv2d(
	input []float64,
	weight []float64,
	bias []float64,
	batch int,
	inChannels int,
	height int,
	width int,
	outChannels int,
	kernelHeight int,
	kernelWidth int,
	strideHeight int,
	strideWidth int,
	padHeight int,
	padWidth int,
	dilationHeight int,
	dilationWidth int,
	groups int,
	heightOut int,
	widthOut int,
) error {
	const operation = "metal.conv2d"

	if batch <= 0 || inChannels <= 0 || height <= 0 || width <= 0 ||
		outChannels <= 0 || kernelHeight <= 0 || kernelWidth <= 0 ||
		strideHeight <= 0 || strideWidth <= 0 ||
		dilationHeight <= 0 || dilationWidth <= 0 || groups <= 0 {
		return fmt.Errorf("%s: invalid dimensions", operation)
	}

	if err := validateMetalConvGroups(
		operation, inChannels, outChannels, groups,
	); err != nil {
		return err
	}

	expectedHeightOut := (height+2*padHeight-dilationHeight*(kernelHeight-1)-1)/strideHeight + 1
	expectedWidthOut := (width+2*padWidth-dilationWidth*(kernelWidth-1)-1)/strideWidth + 1

	return metalConvolutionContract{
		operation:           operation,
		inputLength:         len(input),
		weightLength:        len(weight),
		biasLength:          len(bias),
		expectedInputLength: batch * inChannels * height * width,
		expectedWeight: outChannels * (inChannels / groups) *
			kernelHeight * kernelWidth,
		expectedBias:        outChannels,
		outputShape:         []int{heightOut, widthOut},
		expectedOutputShape: []int{expectedHeightOut, expectedWidthOut},
	}.Validate()
}

func validateMetalConv2dLengths(
	inputLength int,
	weightLength int,
	biasLength int,
	batch int,
	inChannels int,
	height int,
	width int,
	outChannels int,
	kernelHeight int,
	kernelWidth int,
	strideHeight int,
	strideWidth int,
	padHeight int,
	padWidth int,
	dilationHeight int,
	dilationWidth int,
	groups int,
	heightOut int,
	widthOut int,
) error {
	const operation = "metal.conv2d"

	if batch <= 0 || inChannels <= 0 || height <= 0 || width <= 0 ||
		outChannels <= 0 || kernelHeight <= 0 || kernelWidth <= 0 ||
		strideHeight <= 0 || strideWidth <= 0 ||
		dilationHeight <= 0 || dilationWidth <= 0 || groups <= 0 {
		return fmt.Errorf("%s: invalid dimensions", operation)
	}

	if err := validateMetalConvGroups(
		operation, inChannels, outChannels, groups,
	); err != nil {
		return err
	}

	expectedHeightOut := (height+2*padHeight-dilationHeight*(kernelHeight-1)-1)/strideHeight + 1
	expectedWidthOut := (width+2*padWidth-dilationWidth*(kernelWidth-1)-1)/strideWidth + 1

	return metalConvolutionContract{
		operation:           operation,
		inputLength:         inputLength,
		weightLength:        weightLength,
		biasLength:          biasLength,
		expectedInputLength: batch * inChannels * height * width,
		expectedWeight: outChannels * (inChannels / groups) *
			kernelHeight * kernelWidth,
		expectedBias:        outChannels,
		outputShape:         []int{heightOut, widthOut},
		expectedOutputShape: []int{expectedHeightOut, expectedWidthOut},
	}.Validate()
}

func validateMetalConv3d(
	input []float64,
	weight []float64,
	bias []float64,
	batch int,
	inChannels int,
	depth int,
	height int,
	width int,
	outChannels int,
	kernelDepth int,
	kernelHeight int,
	kernelWidth int,
	strideDepth int,
	strideHeight int,
	strideWidth int,
	padDepth int,
	padHeight int,
	padWidth int,
	dilationDepth int,
	dilationHeight int,
	dilationWidth int,
	groups int,
	depthOut int,
	heightOut int,
	widthOut int,
) error {
	const operation = "metal.conv3d"

	if batch <= 0 || inChannels <= 0 || depth <= 0 || height <= 0 || width <= 0 ||
		outChannels <= 0 || kernelDepth <= 0 || kernelHeight <= 0 ||
		kernelWidth <= 0 || strideDepth <= 0 || strideHeight <= 0 ||
		strideWidth <= 0 || dilationDepth <= 0 || dilationHeight <= 0 ||
		dilationWidth <= 0 || groups <= 0 {
		return fmt.Errorf("%s: invalid dimensions", operation)
	}

	if err := validateMetalConvGroups(
		operation, inChannels, outChannels, groups,
	); err != nil {
		return err
	}

	expectedDepthOut := (depth+2*padDepth-dilationDepth*(kernelDepth-1)-1)/strideDepth + 1
	expectedHeightOut := (height+2*padHeight-dilationHeight*(kernelHeight-1)-1)/strideHeight + 1
	expectedWidthOut := (width+2*padWidth-dilationWidth*(kernelWidth-1)-1)/strideWidth + 1

	return metalConvolutionContract{
		operation:           operation,
		inputLength:         len(input),
		weightLength:        len(weight),
		biasLength:          len(bias),
		expectedInputLength: batch * inChannels * depth * height * width,
		expectedWeight: outChannels * (inChannels / groups) *
			kernelDepth * kernelHeight * kernelWidth,
		expectedBias:        outChannels,
		outputShape:         []int{depthOut, heightOut, widthOut},
		expectedOutputShape: []int{expectedDepthOut, expectedHeightOut, expectedWidthOut},
	}.Validate()
}

func validateMetalConvTranspose2d(
	input []float64,
	weight []float64,
	bias []float64,
	batch int,
	inChannels int,
	height int,
	width int,
	outChannels int,
	kernelHeight int,
	kernelWidth int,
	strideHeight int,
	strideWidth int,
	padHeight int,
	padWidth int,
	dilationHeight int,
	dilationWidth int,
	groups int,
	heightOut int,
	widthOut int,
) error {
	const operation = "metal.conv_transpose2d"

	if batch <= 0 || inChannels <= 0 || height <= 0 || width <= 0 ||
		outChannels <= 0 || kernelHeight <= 0 || kernelWidth <= 0 ||
		strideHeight <= 0 || strideWidth <= 0 ||
		dilationHeight <= 0 || dilationWidth <= 0 || groups <= 0 {
		return fmt.Errorf("%s: invalid dimensions", operation)
	}

	if err := validateMetalConvGroups(
		operation, inChannels, outChannels, groups,
	); err != nil {
		return err
	}

	expectedHeightOut := (height-1)*strideHeight - 2*padHeight +
		dilationHeight*(kernelHeight-1) + 1
	expectedWidthOut := (width-1)*strideWidth - 2*padWidth +
		dilationWidth*(kernelWidth-1) + 1

	return metalConvolutionContract{
		operation:           operation,
		inputLength:         len(input),
		weightLength:        len(weight),
		biasLength:          len(bias),
		expectedInputLength: batch * inChannels * height * width,
		expectedWeight: inChannels * (outChannels / groups) *
			kernelHeight * kernelWidth,
		expectedBias:        outChannels,
		outputShape:         []int{heightOut, widthOut},
		expectedOutputShape: []int{expectedHeightOut, expectedWidthOut},
	}.Validate()
}

func validateMetalConvTranspose2dLengths(
	inputLength int,
	weightLength int,
	biasLength int,
	batch int,
	inChannels int,
	height int,
	width int,
	outChannels int,
	kernelHeight int,
	kernelWidth int,
	strideHeight int,
	strideWidth int,
	padHeight int,
	padWidth int,
	dilationHeight int,
	dilationWidth int,
	groups int,
	outPadHeight int,
	outPadWidth int,
	heightOut int,
	widthOut int,
) error {
	const operation = "metal.conv_transpose2d"

	if batch <= 0 || inChannels <= 0 || height <= 0 || width <= 0 ||
		outChannels <= 0 || kernelHeight <= 0 || kernelWidth <= 0 ||
		strideHeight <= 0 || strideWidth <= 0 ||
		dilationHeight <= 0 || dilationWidth <= 0 || groups <= 0 {
		return fmt.Errorf("%s: invalid dimensions", operation)
	}

	if err := validateMetalConvGroups(
		operation, inChannels, outChannels, groups,
	); err != nil {
		return err
	}

	expectedHeightOut := (height-1)*strideHeight - 2*padHeight +
		dilationHeight*(kernelHeight-1) + outPadHeight + 1
	expectedWidthOut := (width-1)*strideWidth - 2*padWidth +
		dilationWidth*(kernelWidth-1) + outPadWidth + 1

	return metalConvolutionContract{
		operation:           operation,
		inputLength:         inputLength,
		weightLength:        weightLength,
		biasLength:          biasLength,
		expectedInputLength: batch * inChannels * height * width,
		expectedWeight: inChannels * (outChannels / groups) *
			kernelHeight * kernelWidth,
		expectedBias:        outChannels,
		outputShape:         []int{heightOut, widthOut},
		expectedOutputShape: []int{expectedHeightOut, expectedWidthOut},
	}.Validate()
}

func validateMetalConvGroups(
	operation string,
	inChannels int,
	outChannels int,
	groups int,
) error {
	if inChannels%groups == 0 && outChannels%groups == 0 {
		return nil
	}

	return fmt.Errorf(
		"%s: groups=%d must divide InC=%d and OutC=%d",
		operation,
		groups,
		inChannels,
		outChannels,
	)
}
