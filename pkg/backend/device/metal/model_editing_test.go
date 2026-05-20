package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestKernelRegistry_MetalModelEditingKernels(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			convey.Convey("Given Metal tensors for model-editing kernels", testingObject, func() {
				runWeightGraftAddParityCase(testingObject, backend, elementCount)
			})
		})
	}
}

func runWeightGraftAddParityCase(
	testingObject testing.TB,
	backend *Backend,
	elementCount int,
) {
	fixture := weightGraftFixtureForTest(elementCount)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	weights := uploadDTypeTensorForTest(testingObject, backend, shape, dtype.Float32, fixture.weightsBytes)
	injection := uploadDTypeTensorForTest(testingObject, backend, shape, dtype.Float32, fixture.injectionBytes)
	defer closeBenchmarkTensors(weights, injection)

	err := lookupWeightGraftAddFloat32Kernel(testingObject).Run(weights, injection)
	convey.So(err, convey.ShouldBeNil)
	assertUtilityBytesForTest(testingObject, backend, weights, dtype.Float32, fixture.expectedBytes)
}

func lookupWeightGraftAddFloat32Kernel(testingObject testing.TB) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("weight_graft_add_float32", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
		Outputs: []dtype.DType{dtype.Float32},
	}, tensor.Metal)

	if !ok {
		testingObject.Fatalf("weight_graft_add_float32 Metal kernel not registered")
	}

	return kernel
}
