package metal

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestKernelRegistry_MetalHawkesKernelMatrix(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalHawkesMarkovDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			for _, elementCount := range parityElementCounts {
				elementCount := elementCount

				testingObject.Run(fmt.Sprintf("N=%d", elementCount), func(testingObject *testing.T) {
					convey.Convey(
						"Given Metal "+storageDType.Name()+" hawkes_kernel_matrix tensors",
						testingObject,
						func() {
							runHawkesKernelMatrixParityCase(
								testingObject, backend, storageDType, elementCount,
							)
						},
					)
				})
			}
		})
	}
}

func TestKernelRegistry_MetalHawkesKernelMatrixSliceRegression(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range []dtype.DType{dtype.Float32, dtype.Float16, dtype.BFloat16} {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			convey.Convey(
				"Given Metal "+storageDType.Name()+" hawkes_intensity regression after hawkes_kernel_matrix",
				testingObject,
				func() {
					runHawkesIntensityParityCase(testingObject, backend, storageDType, 64)
				},
			)
		})
	}
}

func runHawkesKernelMatrixParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	eventCount := hawkesMatrixEventCount(elementCount)
	fixture := hawkesKernelMatrixDTypeBytes(elementCount, storageDType)
	eventShape := mustShapeForTest(testingObject, []int{eventCount})
	outShape := mustShapeForTest(testingObject, []int{eventCount, eventCount})
	events, alpha, beta, out := hawkesKernelMatrixTensorsForTest(
		testingObject, backend, storageDType, eventShape, outShape, fixture,
	)
	defer closeBenchmarkTensors(events, alpha, beta, out)

	err := lookupHawkesKernelMatrixKernel(testingObject, storageDType).Run(events, alpha, beta, out)
	convey.So(err, convey.ShouldBeNil)
	assertHawkesKernelMatrixBytesForTest(testingObject, backend, out, storageDType, fixture)
}

func hawkesKernelMatrixTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	eventShape tensor.Shape,
	outShape tensor.Shape,
	fixture hawkesKernelMatrixFixture,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	scalarShape := scalarShapeForTest(testingObject)
	events := uploadDTypeTensorForTest(testingObject, backend, eventShape, storageDType, fixture.eventBytes)
	alpha := uploadDTypeTensorForTest(testingObject, backend, scalarShape, storageDType, fixture.alphaBytes)
	beta := uploadDTypeTensorForTest(testingObject, backend, scalarShape, storageDType, fixture.betaBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return events, alpha, beta, out
}

func assertHawkesKernelMatrixBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	output tensor.Tensor,
	storageDType dtype.DType,
	fixture hawkesKernelMatrixFixture,
) {
	testingObject.Helper()

	if storageDType == dtype.Float32 {
		expectedValues := decodeDTypeBytesToFloat32(fixture.expectedBytes, storageDType)
		assertFloat32TensorForTest(testingObject, backend, output, expectedValues, hawkesKernelMatrixMaxULP)
		return
	}

	assertDTypeBytesForTest(
		testingObject, backend, output, storageDType, fixture.expectedBytes, hawkesKernelMatrixMaxULP,
	)
}
