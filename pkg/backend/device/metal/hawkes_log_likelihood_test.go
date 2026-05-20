package metal

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestKernelRegistry_MetalHawkesLogLikelihood(testingObject *testing.T) {
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
						"Given Metal "+storageDType.Name()+" hawkes_log_likelihood tensors",
						testingObject,
						func() {
							runHawkesLogLikelihoodParityCase(
								testingObject, backend, storageDType, elementCount,
							)
						},
					)
				})
			}
		})
	}
}

func TestKernelRegistry_MetalHawkesLogLikelihoodSliceRegression(testingObject *testing.T) {
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
				"Given Metal "+storageDType.Name()+" hawkes intensity and kernel_matrix regression after hawkes_log_likelihood",
				testingObject,
				func() {
					runHawkesIntensityParityCase(testingObject, backend, storageDType, 64)
					runHawkesKernelMatrixParityCase(testingObject, backend, storageDType, 64)
				},
			)
		})
	}
}

func runHawkesLogLikelihoodParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := hawkesLogLikelihoodDTypeBytes(elementCount, storageDType)
	eventShape := mustShapeForTest(testingObject, []int{elementCount})
	outShape := scalarShapeForTest(testingObject)
	events, totalTime, baseline, alpha, beta, out := hawkesLogLikelihoodTensorsForTest(
		testingObject, backend, storageDType, eventShape, outShape, fixture,
	)
	defer closeBenchmarkTensors(events, totalTime, baseline, alpha, beta, out)

	err := lookupHawkesLogLikelihoodKernel(testingObject, storageDType).Run(
		events, totalTime, baseline, alpha, beta, out,
	)
	convey.So(err, convey.ShouldBeNil)
	assertHawkesLogLikelihoodBytesForTest(testingObject, backend, out, storageDType, fixture)
}

func hawkesLogLikelihoodTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	eventShape tensor.Shape,
	outShape tensor.Shape,
	fixture hawkesLogLikelihoodFixture,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	scalarShape := scalarShapeForTest(testingObject)
	events := uploadDTypeTensorForTest(testingObject, backend, eventShape, storageDType, fixture.eventBytes)
	totalTime := uploadDTypeTensorForTest(testingObject, backend, scalarShape, storageDType, fixture.totalTimeBytes)
	baseline := uploadDTypeTensorForTest(testingObject, backend, scalarShape, storageDType, fixture.baselineBytes)
	alpha := uploadDTypeTensorForTest(testingObject, backend, scalarShape, storageDType, fixture.alphaBytes)
	beta := uploadDTypeTensorForTest(testingObject, backend, scalarShape, storageDType, fixture.betaBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return events, totalTime, baseline, alpha, beta, out
}

func assertHawkesLogLikelihoodBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	output tensor.Tensor,
	storageDType dtype.DType,
	fixture hawkesLogLikelihoodFixture,
) {
	testingObject.Helper()

	if storageDType == dtype.Float32 {
		expectedValues := decodeDTypeBytesToFloat32(fixture.expectedBytes, storageDType)
		assertFloat32TensorForTest(testingObject, backend, output, expectedValues, hawkesLogLikelihoodMaxULP)
		return
	}

	assertDTypeBytesForTest(
		testingObject, backend, output, storageDType, fixture.expectedBytes, hawkesLogLikelihoodMaxULP,
	)
}
