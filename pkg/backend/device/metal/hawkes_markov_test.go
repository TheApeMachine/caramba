package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestKernelRegistry_MetalHawkesMarkovDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalHawkesMarkovDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalHawkesMarkovDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalHawkesMarkovDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" tensors for Hawkes and Markov kernels", testingObject, func() {
				runHawkesIntensityParityCase(testingObject, backend, storageDType, elementCount)
				runHawkesKernelMatrixParityCase(testingObject, backend, storageDType, elementCount)
				runHawkesLogLikelihoodParityCase(testingObject, backend, storageDType, elementCount)
				runMarkovMutualInformationParityCase(testingObject, backend, storageDType, elementCount)
				runMarkovBlanketPartitionParityCase(testingObject, backend, storageDType, elementCount)
				runMarkovFlowParityCase(testingObject, backend, "markov_flow_active", storageDType, elementCount)
				runMarkovFlowParityCase(testingObject, backend, "markov_flow_internal", storageDType, elementCount)
			})
		})
	}
}

func runHawkesIntensityParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := hawkesIntensityFixtureForTest(storageDType, elementCount, elementCount)
	eventShape := mustShapeForTest(testingObject, []int{elementCount})
	outShape := mustShapeForTest(testingObject, []int{elementCount})
	events, queryTimes, baseline, alpha, beta, out := hawkesFiveTensorsForTest(
		testingObject, backend, storageDType, eventShape, outShape, fixture,
	)
	defer closeBenchmarkTensors(events, queryTimes, baseline, alpha, beta, out)

	err := lookupHawkesIntensityKernel(testingObject, storageDType).Run(events, queryTimes, baseline, alpha, beta, out)
	convey.So(err, convey.ShouldBeNil)
	assertHawkesMarkovBytesForTest(testingObject, backend, out, storageDType, fixture, hawkesScalarULP(storageDType))
}

func runHawkesKernelMatrixParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	eventCount := hawkesMatrixEventCount(elementCount)
	fixture := hawkesKernelMatrixFixtureForTest(storageDType, eventCount)
	eventShape := mustShapeForTest(testingObject, []int{eventCount})
	outShape := mustShapeForTest(testingObject, []int{eventCount, eventCount})
	events := uploadDTypeTensorForTest(testingObject, backend, eventShape, storageDType, fixture.firstBytes)
	alpha := uploadDTypeTensorForTest(testingObject, backend, scalarShapeForTest(testingObject), storageDType, fixture.thirdBytes)
	beta := uploadDTypeTensorForTest(testingObject, backend, scalarShapeForTest(testingObject), storageDType, fixture.fourthBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(events, alpha, beta, out)

	err := lookupHawkesKernelMatrixKernel(testingObject, storageDType).Run(events, alpha, beta, out)
	convey.So(err, convey.ShouldBeNil)
	assertHawkesMarkovBytesForTest(testingObject, backend, out, storageDType, fixture, 3)
}

func runHawkesLogLikelihoodParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := hawkesLogLikelihoodFixtureForTest(storageDType, elementCount)
	eventShape := mustShapeForTest(testingObject, []int{elementCount})
	outShape := scalarShapeForTest(testingObject)
	events, totalTime, baseline, alpha, beta, out := hawkesFiveTensorsForTest(
		testingObject, backend, storageDType, eventShape, outShape, fixture,
	)
	defer closeBenchmarkTensors(events, totalTime, baseline, alpha, beta, out)

	err := lookupHawkesLogLikelihoodKernel(testingObject, storageDType).Run(events, totalTime, baseline, alpha, beta, out)
	convey.So(err, convey.ShouldBeNil)
	assertHawkesMarkovBytesForTest(testingObject, backend, out, storageDType, fixture, hawkesScalarULP(storageDType))
}

func runMarkovMutualInformationParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := markovMutualInformationFixtureForTest(storageDType, markovRowsForTest(elementCount), elementCount)
	inputShape := mustShapeForTest(testingObject, []int{fixture.rows, fixture.cols})
	outShape := scalarShapeForTest(testingObject)
	joint := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, fixture.firstBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(joint, out)

	err := lookupMarkovMutualInformationKernel(testingObject, storageDType).Run(joint, out)
	convey.So(err, convey.ShouldBeNil)
	assertHawkesMarkovBytesForTest(testingObject, backend, out, storageDType, fixture, hawkesScalarULP(storageDType))
}

func runMarkovBlanketPartitionParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	nodeCount := hawkesMatrixEventCount(elementCount)
	fixture := markovPartitionFixtureForTest(storageDType, nodeCount)
	matrixShape := mustShapeForTest(testingObject, []int{nodeCount, nodeCount})
	labelShape := mustShapeForTest(testingObject, []int{len(fixture.labels)})
	outShape := mustShapeForTest(testingObject, []int{nodeCount})
	adjacency := uploadDTypeTensorForTest(testingObject, backend, matrixShape, storageDType, fixture.firstBytes)
	internal := uploadDTypeTensorForTest(testingObject, backend, labelShape, dtype.Int32, int32ValuesToBytes(fixture.labels))
	out := emptyTensorForTest(testingObject, backend, outShape, dtype.Int32)
	defer closeBenchmarkTensors(adjacency, internal, out)

	err := lookupMarkovBlanketPartitionKernel(testingObject, storageDType).Run(adjacency, internal, out)
	convey.So(err, convey.ShouldBeNil)
	assertRawDTypeBytesForTest(testingObject, backend, out, dtype.Int32, int32ValuesToBytes(fixture.expectedLabels))
}

func runMarkovFlowParityCase(
	testingObject testing.TB,
	backend *Backend,
	name string,
	storageDType dtype.DType,
	elementCount int,
) {
	nodeCount := hawkesMatrixEventCount(elementCount)
	fixture := markovFlowFixtureForTest(name, storageDType, nodeCount)
	matrixShape := mustShapeForTest(testingObject, []int{nodeCount, nodeCount})
	labelShape := mustShapeForTest(testingObject, []int{nodeCount})
	mi := uploadDTypeTensorForTest(testingObject, backend, matrixShape, storageDType, fixture.firstBytes)
	partition := uploadDTypeTensorForTest(testingObject, backend, labelShape, dtype.Int32, int32ValuesToBytes(fixture.labels))
	out := emptyTensorForTest(testingObject, backend, labelShape, storageDType)
	defer closeBenchmarkTensors(mi, partition, out)

	err := lookupMarkovFlowKernel(testingObject, name, storageDType).Run(mi, partition, out)
	convey.So(err, convey.ShouldBeNil)
	assertHawkesMarkovBytesForTest(testingObject, backend, out, storageDType, fixture, 2)
}

func hawkesFiveTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	eventShape tensor.Shape,
	outShape tensor.Shape,
	fixture hawkesMarkovFixture,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	scalarShape := scalarShapeForTest(testingObject)
	events := uploadDTypeTensorForTest(testingObject, backend, eventShape, storageDType, fixture.firstBytes)
	second := uploadDTypeTensorForTest(testingObject, backend, outShape, storageDType, fixture.secondBytes)
	baseline := uploadDTypeTensorForTest(testingObject, backend, scalarShape, storageDType, fixture.thirdBytes)
	alpha := uploadDTypeTensorForTest(testingObject, backend, scalarShape, storageDType, fixture.fourthBytes)
	beta := uploadDTypeTensorForTest(testingObject, backend, scalarShape, storageDType, fixture.fifthBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	return events, second, baseline, alpha, beta, out
}

func lookupHawkesIntensityKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	return lookupHawkesMarkovKernel(testingObject, "hawkes_intensity", storageDType, 5, false)
}

func lookupHawkesKernelMatrixKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	return lookupHawkesMarkovKernel(testingObject, "hawkes_kernel_matrix", storageDType, 3, false)
}

func lookupHawkesLogLikelihoodKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	return lookupHawkesMarkovKernel(testingObject, "hawkes_log_likelihood", storageDType, 5, false)
}

func lookupMarkovMutualInformationKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	return lookupHawkesMarkovKernel(testingObject, "markov_mutual_information", storageDType, 1, false)
}

func lookupMarkovBlanketPartitionKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("markov_blanket_partition", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType, dtype.Int32},
		Outputs: []dtype.DType{dtype.Int32},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s markov_blanket_partition kernel", storageDType.Name())
	}

	return kernel
}

func lookupMarkovFlowKernel(testingObject testing.TB, name string, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation(name, kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType, dtype.Int32},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s %s kernel", storageDType.Name(), name)
	}

	return kernel
}

func lookupHawkesMarkovKernel(
	testingObject testing.TB,
	name string,
	storageDType dtype.DType,
	inputCount int,
	intOutput bool,
) kernels.Kernel {
	testingObject.Helper()

	inputs := make([]dtype.DType, inputCount)
	for index := range inputs {
		inputs[index] = storageDType
	}

	outputs := []dtype.DType{storageDType}
	if intOutput {
		outputs = []dtype.DType{dtype.Int32}
	}

	kernel, ok := kernels.Default.LookupLocation(name, kernels.Signature{
		Layout: tensor.LayoutDense, Inputs: inputs, Outputs: outputs,
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s %s kernel", storageDType.Name(), name)
	}

	return kernel
}

func assertHawkesMarkovBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	storageDType dtype.DType,
	fixture hawkesMarkovFixture,
	maxULP uint32,
) {
	testingObject.Helper()

	if storageDType == dtype.Float32 {
		assertFloat32TensorForTest(testingObject, backend, input, fixture.expectedFloat32, maxULP)
		return
	}

	assertDTypeBytesForTest(testingObject, backend, input, storageDType, fixture.expectedBytes, maxULP)
}

func scalarShapeForTest(testingObject testing.TB) tensor.Shape {
	testingObject.Helper()
	return mustShapeForTest(testingObject, []int{1})
}
