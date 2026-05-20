package metal

import (
	"unsafe"

	cpuconvolution "github.com/theapemachine/caramba/pkg/backend/device/cpu/convolution"
	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

const convTranspose2DKernelWidth = 3

type convTranspose2DFixture struct {
	inputBytes    []byte
	weightBytes   []byte
	biasBytes     []byte
	expectedBytes []byte
}

func convTransposeInputWidthForTest(widthCase int) int {
	if widthCase < convTranspose2DKernelWidth {
		return convTranspose2DKernelWidth
	}

	return widthCase
}

func convTranspose2DFixtureForTest(widthCase int, storageDType dtype.DType) convTranspose2DFixture {
	inputWidth := convTransposeInputWidthForTest(widthCase)

	batch, inChannels, outChannels := 2, 2, 3
	inputHeight, kernelHeight, kernelWidth := 3, 2, convTranspose2DKernelWidth
	outputHeight := inputHeight + kernelHeight - 1
	outputWidth := inputWidth + kernelWidth - 1

	inputShape, err := tensor.NewShape([]int{batch, inChannels, inputHeight, inputWidth})
	if err != nil {
		panic(err)
	}

	weightShape, err := tensor.NewShape([]int{inChannels, outChannels, kernelHeight, kernelWidth})
	if err != nil {
		panic(err)
	}

	biasShape, err := tensor.NewShape([]int{outChannels})
	if err != nil {
		panic(err)
	}

	outShape, err := tensor.NewShape([]int{batch, outChannels, outputHeight, outputWidth})
	if err != nil {
		panic(err)
	}

	hostInput, err := tensor.NewZeroed(inputShape, dtype.Float32)
	if err != nil {
		panic(err)
	}

	hostWeight, err := tensor.NewZeroed(weightShape, dtype.Float32)
	if err != nil {
		panic(err)
	}

	hostBias, err := tensor.NewZeroed(biasShape, dtype.Float32)
	if err != nil {
		panic(err)
	}

	hostOutput, err := tensor.NewZeroed(outShape, dtype.Float32)
	if err != nil {
		panic(err)
	}

	inputValues := projectionValues(batch*inChannels*inputHeight*inputWidth, 73, 64)
	weightValues := projectionValues(inChannels*outChannels*kernelHeight*kernelWidth, 43, 128)
	biasValues := projectionValues(outChannels, 23, 32)

	inputView, err := hostInput.Float32Native()
	if err != nil {
		panic(err)
	}

	copy(inputView, inputValues)

	weightView, err := hostWeight.Float32Native()
	if err != nil {
		panic(err)
	}

	copy(weightView, weightValues)

	biasView, err := hostBias.Float32Native()
	if err != nil {
		panic(err)
	}

	copy(biasView, biasValues)

	runConvTranspose2DScalarReference(hostInput, hostWeight, hostBias, hostOutput)

	outputView, err := hostOutput.Float32Native()
	if err != nil {
		panic(err)
	}

	if storageDType == dtype.Float32 {
		return convTranspose2DFixture{
			inputBytes:    dtypeconvert.Float32ToBytes(inputView),
			weightBytes:   dtypeconvert.Float32ToBytes(weightView),
			biasBytes:     dtypeconvert.Float32ToBytes(biasView),
			expectedBytes: dtypeconvert.Float32ToBytes(outputView),
		}
	}

	inputBytes := encodeFloat32ValuesAsDType(inputView, storageDType)
	weightBytes := encodeFloat32ValuesAsDType(weightView, storageDType)
	biasBytes := encodeFloat32ValuesAsDType(biasView, storageDType)

	storedInput := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	storedWeight := decodeDTypeBytesToFloat32(weightBytes, storageDType)
	storedBias := decodeDTypeBytesToFloat32(biasBytes, storageDType)

	roundTripInput, err := tensor.NewZeroed(inputShape, dtype.Float32)
	if err != nil {
		panic(err)
	}

	roundTripWeight, err := tensor.NewZeroed(weightShape, dtype.Float32)
	if err != nil {
		panic(err)
	}

	roundTripBias, err := tensor.NewZeroed(biasShape, dtype.Float32)
	if err != nil {
		panic(err)
	}

	roundTripInputView, err := roundTripInput.Float32Native()
	if err != nil {
		panic(err)
	}

	copy(roundTripInputView, storedInput)

	roundTripWeightView, err := roundTripWeight.Float32Native()
	if err != nil {
		panic(err)
	}

	copy(roundTripWeightView, storedWeight)

	roundTripBiasView, err := roundTripBias.Float32Native()
	if err != nil {
		panic(err)
	}

	copy(roundTripBiasView, storedBias)

	runConvTranspose2DScalarReference(roundTripInput, roundTripWeight, roundTripBias, hostOutput)

	outputView, err = hostOutput.Float32Native()
	if err != nil {
		panic(err)
	}

	return convTranspose2DFixture{
		inputBytes:    inputBytes,
		weightBytes:   weightBytes,
		biasBytes:     biasBytes,
		expectedBytes: encodeFloat32ValuesAsDType(outputView, storageDType),
	}
}

func runConvTranspose2DScalarReference(
	hostInput tensor.Tensor,
	hostWeight tensor.Tensor,
	hostBias tensor.Tensor,
	hostOutput tensor.Tensor,
) {
	inputView, err := hostInput.Float32Native()
	if err != nil {
		panic(err)
	}

	weightView, err := hostWeight.Float32Native()
	if err != nil {
		panic(err)
	}

	biasView, err := hostBias.Float32Native()
	if err != nil {
		panic(err)
	}

	outputView, err := hostOutput.Float32Native()
	if err != nil {
		panic(err)
	}

	inDims := hostInput.Shape().Dims()
	weightDims := hostWeight.Shape().Dims()
	outDims := hostOutput.Shape().Dims()

	config := cpuconvolution.DefaultConv2DConfig()
	cpuconvolution.ConvTranspose2DFloat32Scalar(
		config,
		unsafe.Pointer(unsafe.SliceData(inputView)),
		unsafe.Pointer(unsafe.SliceData(weightView)),
		unsafe.Pointer(unsafe.SliceData(biasView)),
		unsafe.Pointer(unsafe.SliceData(outputView)),
		inDims[0], inDims[1], inDims[2], inDims[3],
		weightDims[1], weightDims[2], weightDims[3],
		outDims[2], outDims[3],
	)
}

func convTranspose2DDTypeBytes(widthCase int, storageDType dtype.DType) ([]byte, []byte, []byte, []byte) {
	fixture := convTranspose2DFixtureForTest(widthCase, storageDType)

	return fixture.inputBytes, fixture.weightBytes, fixture.biasBytes, fixture.expectedBytes
}
