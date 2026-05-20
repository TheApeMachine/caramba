package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestKernelRegistry_MetalVisionDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalVisionDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalVisionDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalVisionDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, outputWidth := range parityElementCounts {
		outputWidth := outputWidth

		testingObject.Run(testNameForElementCount(outputWidth), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" vision tensors", testingObject, func() {
				runConv2DParityCase(testingObject, backend, storageDType, outputWidth)
			})
		})
	}
}

func runConv2DParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	outputWidth int,
) {
	inputBytes, weightBytes, biasBytes, expectedBytes := conv2DDTypeBytes(outputWidth, storageDType)
	input, weight, bias, out := conv2DTensorsForTest(
		testingObject, backend, outputWidth, storageDType, inputBytes, weightBytes, biasBytes,
	)
	defer closeBenchmarkTensors(input, weight, bias, out)

	err := lookupVisionConv2DKernel(testingObject, storageDType).Run(input, weight, bias, out)
	convey.So(err, convey.ShouldBeNil)
	assertProjectionBytesForTest(testingObject, backend, out, storageDType, expectedBytes)
}

func lookupVisionConv2DKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("conv2d", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType, storageDType, storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s conv2d kernel", storageDType.Name())
	}

	return kernel
}

func lookupVisionPool2DKernel(
	testingObject testing.TB,
	name string,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation(name, kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s %s kernel", storageDType.Name(), name)
	}

	return kernel
}

func conv2DTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	outputWidth int,
	storageDType dtype.DType,
	inputBytes []byte,
	weightBytes []byte,
	biasBytes []byte,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	batch, inChannels, outChannels := 2, 2, 3
	inputHeight, kernelHeight, kernelWidth := 4, 2, 3
	outputHeight := inputHeight - kernelHeight + 1
	inputWidth := outputWidth + kernelWidth - 1
	inputShape := mustShapeForTest(testingObject, []int{batch, inChannels, inputHeight, inputWidth})
	weightShape := mustShapeForTest(testingObject, []int{outChannels, inChannels, kernelHeight, kernelWidth})
	biasShape := mustShapeForTest(testingObject, []int{outChannels})
	outShape := mustShapeForTest(testingObject, []int{batch, outChannels, outputHeight, outputWidth})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	weight := uploadDTypeTensorForTest(testingObject, backend, weightShape, storageDType, weightBytes)
	bias := uploadDTypeTensorForTest(testingObject, backend, biasShape, storageDType, biasBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return input, weight, bias, out
}

func pool2DTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	outputWidth int,
	storageDType dtype.DType,
	inputBytes []byte,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	batch, channels, inputHeight := 2, 3, 4
	inputWidth := outputWidth * 2
	inputShape := mustShapeForTest(testingObject, []int{batch, channels, inputHeight, inputWidth})
	outShape := mustShapeForTest(testingObject, []int{batch, channels, inputHeight / 2, outputWidth})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	maxOut := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	avgOut := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return input, maxOut, avgOut
}

func adaptivePool2DTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	outputWidth int,
	storageDType dtype.DType,
	inputBytes []byte,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	batch, channels, inputHeight := 2, 2, 5
	inputWidth := outputWidth + 3
	inputShape := mustShapeForTest(testingObject, []int{batch, channels, inputHeight, inputWidth})
	outShape := mustShapeForTest(testingObject, []int{batch, channels, 3, outputWidth})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	avgOut := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	maxOut := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return input, avgOut, maxOut
}
