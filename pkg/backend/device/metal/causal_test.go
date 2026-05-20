package metal

import (
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestKernelRegistry_MetalCausalDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalCausalDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalCausalDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalCausalDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" causal tensors", testingObject, func() {
				runBackdoorParityCase(testingObject, backend, storageDType, elementCount)
				runFrontdoorParityCase(testingObject, backend, storageDType, elementCount)
				runDoInterveneParityCase(testingObject, backend, storageDType, elementCount)
				runCATEParityCase(testingObject, backend, storageDType, elementCount)
				runCounterfactualParityCase(testingObject, backend, storageDType, elementCount)
				runIVEstimateParityCase(testingObject, backend, storageDType, causalSampleCount(elementCount))
				runDAGMarkovFactorizationParityCase(testingObject, backend, storageDType, elementCount)
			})
		})
	}
}

func runBackdoorParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	zCount int,
) {
	xCount, yCount := 3, 5
	fixture := backdoorFixtureForTest(xCount, zCount, yCount, storageDType)
	conditional, marginal, out := causalBinaryTensorsForTest(
		testingObject, backend, []int{xCount, zCount, yCount}, []int{zCount},
		[]int{xCount, yCount}, storageDType, fixture,
	)
	defer closeBenchmarkTensors(conditional, marginal, out)

	err := lookupCausalBinaryKernel(testingObject, "backdoor_adjustment", storageDType).Run(
		conditional, marginal, out,
	)
	convey.So(err, convey.ShouldBeNil)
	assertCausalOutput(testingObject, backend, out, storageDType, fixture.expectedBytes, fixture.expectedFloat32)
}

func runFrontdoorParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	mCount int,
) {
	xCount, yCount := 3, 4
	fixture := frontdoorFixtureForTest(xCount, mCount, yCount, storageDType)
	mediator, outcome, marginal, out := causalTernaryTensorsForTest(
		testingObject, backend, []int{xCount, mCount}, []int{xCount, mCount, yCount},
		[]int{xCount}, []int{xCount, yCount}, storageDType, fixture,
	)
	defer closeBenchmarkTensors(mediator, outcome, marginal, out)

	err := lookupCausalTernaryKernel(testingObject, "frontdoor_adjustment", storageDType).Run(
		mediator, outcome, marginal, out,
	)
	convey.So(err, convey.ShouldBeNil)
	assertCausalOutput(testingObject, backend, out, storageDType, fixture.expectedBytes, fixture.expectedFloat32)
}

func runDoInterveneParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	nodeCount := causalMatrixSizeForTest(elementCount)
	fixture, _ := doInterveneFixtureForTest(nodeCount, storageDType)
	adjacency := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{nodeCount, nodeCount}),
		storageDType, fixture.leftBytes,
	)
	intervened := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{len(fixture.rightBytes) / 4}),
		dtype.Int32, fixture.rightBytes,
	)
	out := emptyTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{nodeCount, nodeCount}),
		storageDType,
	)
	defer closeBenchmarkTensors(adjacency, intervened, out)

	err := lookupCausalInt32Kernel(testingObject, "do_intervene", storageDType).Run(adjacency, intervened, out)
	convey.So(err, convey.ShouldBeNil)
	assertCausalOutput(testingObject, backend, out, storageDType, fixture.expectedBytes, fixture.expectedFloat32)
}

func runCATEParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := cateFixtureForTest(elementCount, storageDType)
	treated, control, out := causalBinaryTensorsForTest(
		testingObject, backend, []int{elementCount}, []int{elementCount},
		[]int{elementCount}, storageDType, fixture,
	)
	defer closeBenchmarkTensors(treated, control, out)

	err := lookupCausalBinaryKernel(testingObject, "cate", storageDType).Run(treated, control, out)
	convey.So(err, convey.ShouldBeNil)
	assertCausalOutput(testingObject, backend, out, storageDType, fixture.expectedBytes, fixture.expectedFloat32)
}

func runCounterfactualParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := counterfactualFixtureForTest(elementCount, storageDType)
	observedY, observedX, counterfactualX, slope, out := counterfactualTensorsForTest(
		testingObject, backend, elementCount, storageDType, fixture,
	)
	defer closeBenchmarkTensors(observedY, observedX, counterfactualX, slope, out)

	err := lookupCausalQuaternaryKernel(testingObject, "counterfactual", storageDType).Run(
		observedY, observedX, counterfactualX, slope, out,
	)
	convey.So(err, convey.ShouldBeNil)
	assertCausalOutput(testingObject, backend, out, storageDType, fixture.expectedBytes, fixture.expectedFloat32)
}

func runIVEstimateParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := ivFixtureForTest(elementCount, storageDType)
	instrument, treatment, outcome, out := causalTernaryTensorsForTest(
		testingObject, backend, []int{elementCount}, []int{elementCount},
		[]int{elementCount}, []int{1}, storageDType, fixture,
	)
	defer closeBenchmarkTensors(instrument, treatment, outcome, out)

	err := lookupCausalTernaryKernel(testingObject, "iv_estimate", storageDType).Run(
		instrument, treatment, outcome, out,
	)
	convey.So(err, convey.ShouldBeNil)
	assertCausalOutput(testingObject, backend, out, storageDType, fixture.expectedBytes, fixture.expectedFloat32)
}

func runDAGMarkovFactorizationParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture, parents := dagFixtureForTest(elementCount, storageDType)
	conditionals := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{elementCount}),
		storageDType, fixture.inputBytes,
	)
	parentTensor := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{len(parents)}),
		dtype.Int32, int32ValuesToBytes(parents),
	)
	out := emptyTensorForTest(testingObject, backend, mustShapeForTest(testingObject, []int{1}), storageDType)
	defer closeBenchmarkTensors(conditionals, parentTensor, out)

	err := lookupCausalInt32Kernel(testingObject, "dag_markov_factorization", storageDType).Run(
		conditionals, parentTensor, out,
	)
	convey.So(err, convey.ShouldBeNil)
	assertCausalOutput(testingObject, backend, out, storageDType, fixture.expectedBytes, fixture.expectedFloat32)
}

func causalBinaryTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	leftDims []int,
	rightDims []int,
	outDims []int,
	storageDType dtype.DType,
	fixture causalBinaryFixture,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	left := uploadDTypeTensorForTest(testingObject, backend, mustShapeForTest(testingObject, leftDims), storageDType, fixture.leftBytes)
	right := uploadDTypeTensorForTest(testingObject, backend, mustShapeForTest(testingObject, rightDims), storageDType, fixture.rightBytes)
	out := emptyTensorForTest(testingObject, backend, mustShapeForTest(testingObject, outDims), storageDType)
	return left, right, out
}

func causalTernaryTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	firstDims []int,
	secondDims []int,
	thirdDims []int,
	outDims []int,
	storageDType dtype.DType,
	fixture causalTernaryFixture,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	first := uploadDTypeTensorForTest(testingObject, backend, mustShapeForTest(testingObject, firstDims), storageDType, fixture.firstBytes)
	second := uploadDTypeTensorForTest(testingObject, backend, mustShapeForTest(testingObject, secondDims), storageDType, fixture.secondBytes)
	third := uploadDTypeTensorForTest(testingObject, backend, mustShapeForTest(testingObject, thirdDims), storageDType, fixture.thirdBytes)
	out := emptyTensorForTest(testingObject, backend, mustShapeForTest(testingObject, outDims), storageDType)
	return first, second, third, out
}

func counterfactualTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	elementCount int,
	storageDType dtype.DType,
	fixture causalCounterfactualFixture,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	shape := mustShapeForTest(testingObject, []int{elementCount})
	observedY := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.observedYBytes)
	observedX := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.observedXBytes)
	counterfactualX := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.counterfactualXBytes)
	slope := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{1}), storageDType, fixture.slopeBytes,
	)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	return observedY, observedX, counterfactualX, slope, out
}

func assertCausalOutput(
	testingObject testing.TB,
	backend *Backend,
	out tensor.Tensor,
	storageDType dtype.DType,
	expectedBytes []byte,
	expectedFloat32 []float32,
) {
	testingObject.Helper()

	if storageDType != dtype.Float32 {
		assertDTypeBytesForTest(testingObject, backend, out, storageDType, expectedBytes, 2)
		return
	}

	assertFloat32TensorForTest(testingObject, backend, out, expectedFloat32, 64)
}

func lookupCausalBinaryKernel(testingObject testing.TB, name string, storageDType dtype.DType) kernels.Kernel {
	return lookupCausalKernel(testingObject, name, []dtype.DType{storageDType, storageDType}, storageDType)
}

func lookupCausalTernaryKernel(testingObject testing.TB, name string, storageDType dtype.DType) kernels.Kernel {
	return lookupCausalKernel(
		testingObject, name, []dtype.DType{storageDType, storageDType, storageDType}, storageDType,
	)
}

func lookupCausalQuaternaryKernel(testingObject testing.TB, name string, storageDType dtype.DType) kernels.Kernel {
	return lookupCausalKernel(
		testingObject, name,
		[]dtype.DType{storageDType, storageDType, storageDType, storageDType}, storageDType,
	)
}

func lookupCausalInt32Kernel(testingObject testing.TB, name string, storageDType dtype.DType) kernels.Kernel {
	return lookupCausalKernel(testingObject, name, []dtype.DType{storageDType, dtype.Int32}, storageDType)
}

func lookupCausalKernel(
	testingObject testing.TB,
	name string,
	inputs []dtype.DType,
	output dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation(name, kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  inputs,
		Outputs: []dtype.DType{output},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s %s kernel", output.Name(), name)
	}

	return kernel
}

func causalSampleCount(elementCount int) int {
	return max(2, elementCount)
}

func causalMatrixSizeForTest(elementCount int) int {
	return max(2, int(math.Ceil(math.Sqrt(float64(elementCount)))))
}
