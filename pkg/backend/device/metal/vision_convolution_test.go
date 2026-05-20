package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestKernelRegistry_MetalConvolutionDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalVisionDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalConvolutionDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalConvolutionDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, width := range parityElementCounts {
		width := width

		testingObject.Run(testNameForElementCount(width), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" convolution tensors", testingObject, func() {
				runConv1DParityCase(testingObject, backend, storageDType, width)
				runConv3DParityCase(testingObject, backend, storageDType, width)
				runConvTranspose2DParityCase(testingObject, backend, storageDType, width)
			})
		})
	}
}

func runConv1DParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	width int,
) {
	inputBytes, weightBytes, biasBytes, expectedBytes := conv1DDTypeBytes(width, storageDType)
	input, weight, bias, out := conv1DTensorsForTest(
		testingObject, backend, width, storageDType, inputBytes, weightBytes, biasBytes,
	)
	defer closeBenchmarkTensors(input, weight, bias, out)

	err := lookupVisionConvolutionKernel(testingObject, "conv1d", storageDType).Run(
		input, weight, bias, out,
	)
	convey.So(err, convey.ShouldBeNil)
	assertProjectionBytesForTest(testingObject, backend, out, storageDType, expectedBytes)
}

func runConv3DParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	width int,
) {
	inputBytes, weightBytes, biasBytes, expectedBytes := conv3DDTypeBytes(width, storageDType)
	input, weight, bias, out := conv3DTensorsForTest(
		testingObject, backend, width, storageDType, inputBytes, weightBytes, biasBytes,
	)
	defer closeBenchmarkTensors(input, weight, bias, out)

	err := lookupVisionConvolutionKernel(testingObject, "conv3d", storageDType).Run(
		input, weight, bias, out,
	)
	convey.So(err, convey.ShouldBeNil)
	assertProjectionBytesForTest(testingObject, backend, out, storageDType, expectedBytes)
}

func runConvTranspose2DParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	width int,
) {
	inputWidth := convTransposeInputWidthForTest(width)
	inputBytes, weightBytes, biasBytes, expectedBytes :=
		convTranspose2DDTypeBytes(width, storageDType)
	input, weight, bias, out := convTranspose2DTensorsForTest(
		testingObject, backend, inputWidth, storageDType, inputBytes, weightBytes, biasBytes,
	)
	defer closeBenchmarkTensors(input, weight, bias, out)

	err := lookupVisionConvolutionKernel(testingObject, "conv_transpose2d", storageDType).Run(
		input, weight, bias, out,
	)
	convey.So(err, convey.ShouldBeNil)
	assertProjectionBytesForTest(testingObject, backend, out, storageDType, expectedBytes)
}

func lookupVisionConvolutionKernel(
	testingObject testing.TB,
	name string,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation(name, kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType, storageDType, storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s %s kernel", storageDType.Name(), name)
	}

	return kernel
}

func conv1DTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	width int,
	storageDType dtype.DType,
	inputBytes []byte,
	weightBytes []byte,
	biasBytes []byte,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	batch, inChannels, outChannels, kernelLength := 2, 2, 3, 3
	inputLength := width + kernelLength - 1
	inputShape := mustShapeForTest(testingObject, []int{batch, inChannels, inputLength})
	weightShape := mustShapeForTest(testingObject, []int{outChannels, inChannels, kernelLength})
	biasShape := mustShapeForTest(testingObject, []int{outChannels})
	outShape := mustShapeForTest(testingObject, []int{batch, outChannels, width})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	weight := uploadDTypeTensorForTest(testingObject, backend, weightShape, storageDType, weightBytes)
	bias := uploadDTypeTensorForTest(testingObject, backend, biasShape, storageDType, biasBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return input, weight, bias, out
}

func conv3DTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	width int,
	storageDType dtype.DType,
	inputBytes []byte,
	weightBytes []byte,
	biasBytes []byte,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	batch, inChannels, outChannels := 1, 2, 2
	inputDepth, inputHeight, kernelDepth, kernelHeight, kernelWidth := 3, 3, 2, 2, 3
	inputWidth := width + kernelWidth - 1
	inputShape := mustShapeForTest(
		testingObject, []int{batch, inChannels, inputDepth, inputHeight, inputWidth},
	)
	weightShape := mustShapeForTest(
		testingObject, []int{outChannels, inChannels, kernelDepth, kernelHeight, kernelWidth},
	)
	biasShape := mustShapeForTest(testingObject, []int{outChannels})
	outShape := mustShapeForTest(testingObject, []int{batch, outChannels, 2, 2, width})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	weight := uploadDTypeTensorForTest(testingObject, backend, weightShape, storageDType, weightBytes)
	bias := uploadDTypeTensorForTest(testingObject, backend, biasShape, storageDType, biasBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return input, weight, bias, out
}

func convTranspose2DTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	width int,
	storageDType dtype.DType,
	inputBytes []byte,
	weightBytes []byte,
	biasBytes []byte,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	batch, inChannels, outChannels := 2, 2, 3
	inputHeight, kernelHeight, kernelWidth := 3, 2, 3
	inputShape := mustShapeForTest(testingObject, []int{batch, inChannels, inputHeight, width})
	weightShape := mustShapeForTest(testingObject, []int{inChannels, outChannels, kernelHeight, kernelWidth})
	biasShape := mustShapeForTest(testingObject, []int{outChannels})
	outShape := mustShapeForTest(testingObject, []int{batch, outChannels, 4, width + 2})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	weight := uploadDTypeTensorForTest(testingObject, backend, weightShape, storageDType, weightBytes)
	bias := uploadDTypeTensorForTest(testingObject, backend, biasShape, storageDType, biasBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return input, weight, bias, out
}
