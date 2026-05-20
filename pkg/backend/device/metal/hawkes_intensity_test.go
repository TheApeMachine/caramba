package metal

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestKernelRegistry_MetalHawkesIntensity(testingObject *testing.T) {
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
						"Given Metal "+storageDType.Name()+" hawkes_intensity tensors",
						testingObject,
						func() {
							runHawkesIntensityParityCase(
								testingObject, backend, storageDType, elementCount,
							)
						},
					)
				})
			}
		})
	}
}

func TestKernelRegistry_MetalHawkesIntensitySliceRegression(testingObject *testing.T) {
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
				"Given Metal "+storageDType.Name()+" adaptive_max_pool2d regression after hawkes_intensity",
				testingObject,
				func() {
					runAdaptiveMaxPool2DParityCase(testingObject, backend, storageDType, 64)
				},
			)
		})
	}
}

func runHawkesIntensityParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := hawkesIntensityDTypeBytes(elementCount, storageDType)
	eventShape := mustShapeForTest(testingObject, []int{elementCount})
	outShape := mustShapeForTest(testingObject, []int{elementCount})
	events, queryTimes, baseline, alpha, beta, out := hawkesIntensityTensorsForTest(
		testingObject, backend, storageDType, eventShape, outShape, fixture,
	)
	defer closeBenchmarkTensors(events, queryTimes, baseline, alpha, beta, out)

	err := lookupHawkesIntensityKernel(testingObject, storageDType).Run(
		events, queryTimes, baseline, alpha, beta, out,
	)
	convey.So(err, convey.ShouldBeNil)
	assertHawkesIntensityBytesForTest(testingObject, backend, out, storageDType, fixture)
}

func hawkesIntensityTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	eventShape tensor.Shape,
	outShape tensor.Shape,
	fixture hawkesIntensityFixture,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	scalarShape := scalarShapeForTest(testingObject)
	events := uploadDTypeTensorForTest(testingObject, backend, eventShape, storageDType, fixture.eventBytes)
	queryTimes := uploadDTypeTensorForTest(testingObject, backend, outShape, storageDType, fixture.queryBytes)
	baseline := uploadDTypeTensorForTest(testingObject, backend, scalarShape, storageDType, fixture.baselineBytes)
	alpha := uploadDTypeTensorForTest(testingObject, backend, scalarShape, storageDType, fixture.alphaBytes)
	beta := uploadDTypeTensorForTest(testingObject, backend, scalarShape, storageDType, fixture.betaBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return events, queryTimes, baseline, alpha, beta, out
}

func assertHawkesIntensityBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	output tensor.Tensor,
	storageDType dtype.DType,
	fixture hawkesIntensityFixture,
) {
	testingObject.Helper()

	if storageDType == dtype.Float32 {
		expectedValues := decodeDTypeBytesToFloat32(fixture.expectedBytes, storageDType)
		assertFloat32TensorForTest(testingObject, backend, output, expectedValues, hawkesIntensityMaxULP)
		return
	}

	assertDTypeBytesForTest(
		testingObject, backend, output, storageDType, fixture.expectedBytes, hawkesIntensityMaxULP,
	)
}
