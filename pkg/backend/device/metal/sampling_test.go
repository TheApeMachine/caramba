package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestKernelRegistry_MetalSamplingDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalSamplingDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalSamplingDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalSamplingDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" tensors for sampling", testingObject, func() {
				runGreedySamplingParityCase(testingObject, backend, storageDType, elementCount)
				runDrawSamplingParityCase(testingObject, backend, storageDType, elementCount, "topk_sample")
				runDrawSamplingParityCase(testingObject, backend, storageDType, elementCount, "topp_sample")
			})
		})
	}
}

func runGreedySamplingParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := greedySamplingFixtureForTest(elementCount, storageDType)
	actual := runSamplingKernelForTest(
		testingObject, backend, storageDType, elementCount, "greedy_sample", fixture.inputBytes,
	)
	convey.So(actual, convey.ShouldEqual, fixture.expected)
}

func runDrawSamplingParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
	name string,
) {
	fixture := drawSamplingFixtureForTest(elementCount, storageDType)
	actual := runSamplingKernelForTest(
		testingObject, backend, storageDType, elementCount, name, fixture.inputBytes,
	)
	convey.So(actual, convey.ShouldEqual, fixture.expected)
}

func runSamplingKernelForTest(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
	name string,
	inputBytes []byte,
) int32 {
	testingObject.Helper()

	shape := mustShapeForTest(testingObject, []int{elementCount})
	outShape := mustShapeForTest(testingObject, []int{1})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, inputBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, dtype.Int32)
	defer closeBenchmarkTensors(input, out)

	err := lookupSamplingKernel(testingObject, name, storageDType).Run(input, out)
	convey.So(err, convey.ShouldBeNil)

	return downloadInt32ScalarForTest(testingObject, backend, out)
}

func lookupSamplingKernel(
	testingObject testing.TB,
	name string,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation(name, kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType},
		Outputs: []dtype.DType{dtype.Int32},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s sampling kernel for %s", storageDType.Name(), name)
	}

	return kernel
}
