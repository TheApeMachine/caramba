package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestKernelRegistry_MetalDropoutDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalDropoutDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalDropoutDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalDropoutDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			convey.Convey("Given a Metal "+storageDType.Name()+" tensor for dropout", testingObject, func() {
				runDropoutParityCase(testingObject, backend, storageDType, elementCount)
			})
		})
	}
}

func runDropoutParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := dropoutFixtureForTest(elementCount, storageDType)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.inputBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	defer closeBenchmarkTensors(input, out)

	err := lookupDropoutKernel(testingObject, storageDType).Run(input, out)
	convey.So(err, convey.ShouldBeNil)
	assertDropoutBytesForTest(testingObject, backend, out, storageDType, fixture)
}

func lookupDropoutKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("dropout", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s dropout kernel", storageDType.Name())
	}

	return kernel
}

func assertDropoutBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	storageDType dtype.DType,
	fixture dropoutFixture,
) {
	testingObject.Helper()

	if storageDType != dtype.Float32 {
		assertDTypeBytesForTest(testingObject, backend, input, storageDType, fixture.expectedBytes, 1)
		return
	}

	assertFloat32TensorForTest(testingObject, backend, input, fixture.expectedFloat32, 0)
}
