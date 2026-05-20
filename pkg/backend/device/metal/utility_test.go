package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestKernelRegistry_MetalUtilityKernels(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			convey.Convey("Given Metal tensors for utility kernels", testingObject, func() {
				runCheckpointParityCase(testingObject, backend, elementCount)
				runTokenizerPackParityCase(testingObject, backend, elementCount)
				runWeightFreezeMaskParityCases(testingObject, backend, elementCount)
			})
		})
	}
}

func runCheckpointParityCase(
	testingObject testing.TB,
	backend *Backend,
	elementCount int,
) {
	fixture := checkpointFixtureForTest(elementCount)
	inputShape := mustShapeForTest(testingObject, []int{elementCount})
	encodedShape := mustShapeForTest(testingObject, []int{len(fixture.encodedBytes)})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, dtype.Float32, fixture.inputBytes)
	encoded := emptyTensorForTest(testingObject, backend, encodedShape, dtype.Uint8)
	decoded := emptyTensorForTest(testingObject, backend, inputShape, dtype.Float32)
	defer closeBenchmarkTensors(input, encoded, decoded)

	encodeErr := lookupCheckpointEncodeKernel(testingObject).Run(input, encoded)
	convey.So(encodeErr, convey.ShouldBeNil)
	assertUtilityBytesForTest(testingObject, backend, encoded, dtype.Uint8, fixture.encodedBytes)

	decodeErr := lookupCheckpointDecodeKernel(testingObject).Run(encoded, decoded)
	convey.So(decodeErr, convey.ShouldBeNil)
	assertUtilityBytesForTest(testingObject, backend, decoded, dtype.Float32, fixture.inputBytes)
}

func runTokenizerPackParityCase(
	testingObject testing.TB,
	backend *Backend,
	elementCount int,
) {
	fixture := tokenizerFixtureForTest(elementCount)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, dtype.Int32, fixture.inputBytes)
	out := emptyTensorForTest(testingObject, backend, shape, dtype.Int32)
	defer closeBenchmarkTensors(input, out)

	err := lookupTokenizerPackKernel(testingObject).Run(input, out)
	convey.So(err, convey.ShouldBeNil)
	assertUtilityBytesForTest(testingObject, backend, out, dtype.Int32, fixture.inputBytes)
}

func runWeightFreezeMaskParityCases(
	testingObject testing.TB,
	backend *Backend,
	elementCount int,
) {
	for _, storageDType := range metalUtilityFloatDTypes {
		runWeightFreezeMaskParityCase(testingObject, backend, elementCount, storageDType)
	}
}

func runWeightFreezeMaskParityCase(
	testingObject testing.TB,
	backend *Backend,
	elementCount int,
	storageDType dtype.DType,
) {
	fixture := weightFreezeFixtureForTest(elementCount, storageDType)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	mask := uploadDTypeTensorForTest(testingObject, backend, shape, dtype.Bool, fixture.maskBytes)
	gradients := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.gradientBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	defer closeBenchmarkTensors(mask, gradients, out)

	err := lookupWeightFreezeMaskKernel(testingObject, storageDType).Run(mask, gradients, out)
	convey.So(err, convey.ShouldBeNil)
	assertDTypeBytesForTest(testingObject, backend, out, storageDType, fixture.expectedBytes, 0)
}

func lookupCheckpointEncodeKernel(testingObject testing.TB) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("checkpoint_encode_float32", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{dtype.Float32},
		Outputs: []dtype.DType{dtype.Uint8},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal checkpoint_encode_float32 kernel")
	}

	return kernel
}

func lookupCheckpointDecodeKernel(testingObject testing.TB) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("checkpoint_decode_float32", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{dtype.Uint8},
		Outputs: []dtype.DType{dtype.Float32},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal checkpoint_decode_float32 kernel")
	}

	return kernel
}

func lookupTokenizerPackKernel(testingObject testing.TB) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("tokenizer_pack_int32", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{dtype.Int32},
		Outputs: []dtype.DType{dtype.Int32},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal tokenizer_pack_int32 kernel")
	}

	return kernel
}

func lookupWeightFreezeMaskKernel(
	testingObject testing.TB,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("weight_freeze_mask", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{dtype.Bool, storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s weight_freeze_mask kernel", storageDType.Name())
	}

	return kernel
}

func assertUtilityBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	expectedDType dtype.DType,
	expectedBytes []byte,
) {
	testingObject.Helper()

	actualDType, actualBytes, err := backend.Download(input)
	if err != nil {
		testingObject.Fatalf("Download failed: %v", err)
	}

	if actualDType != expectedDType {
		testingObject.Fatalf("download dtype mismatch: got %s want %s", actualDType, expectedDType)
	}

	convey.So(actualBytes, convey.ShouldResemble, expectedBytes)
}
