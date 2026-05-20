package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

func TestKernelRegistry_MetalResearchDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalResearchDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalResearchDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalResearchDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" tensors for research kernels", testingObject, func() {
				runVSAResearchParityCases(testingObject, backend, storageDType, elementCount)
				runPredictiveCodingParityCases(testingObject, backend, storageDType, elementCount)
			})
		})
	}
}

func runVSAResearchParityCases(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	runVSABinaryParityCase(testingObject, backend, "vsa_bind", storageDType, elementCount)
	runVSABinaryParityCase(testingObject, backend, "vsa_bundle", storageDType, elementCount)
	runVSAUnaryParityCase(testingObject, backend, "vsa_permute", storageDType, elementCount)
	runVSAUnaryParityCase(testingObject, backend, "vsa_inverse_permute", storageDType, elementCount)
}

func runPredictiveCodingParityCases(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	inCount int,
) {
	outCount := 7
	runPCPredictionParityCase(testingObject, backend, storageDType, outCount, inCount)
	runPCPredictionErrorParityCase(testingObject, backend, storageDType, inCount)
	runPCUpdateRepresentationParityCase(testingObject, backend, storageDType, outCount, inCount)
	runPCUpdateWeightsParityCase(testingObject, backend, storageDType, outCount, inCount)
}

func runVSABinaryParityCase(
	testingObject testing.TB,
	backend *Backend,
	name string,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := vsaBinaryFixtureForTest(name, storageDType, elementCount)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	left := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.leftBytes)
	right := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.rightBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	defer closeBenchmarkTensors(left, right, out)

	err := lookupResearchBinaryKernel(testingObject, name, storageDType).Run(left, right, out)
	convey.So(err, convey.ShouldBeNil)
	assertResearchBytesForTest(testingObject, backend, out, storageDType, fixture.expectedValues, 2)
}

func runVSAUnaryParityCase(
	testingObject testing.TB,
	backend *Backend,
	name string,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := vsaUnaryFixtureForTest(name, storageDType, elementCount)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.inputBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	defer closeBenchmarkTensors(input, out)

	err := lookupResearchUnaryKernel(testingObject, name, storageDType).Run(input, out)
	convey.So(err, convey.ShouldBeNil)
	assertResearchBytesForTest(testingObject, backend, out, storageDType, fixture.expectedValues, 0)
}

func runPCPredictionParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	outCount int,
	inCount int,
) {
	fixture := pcFixtureForTest(storageDType, outCount, inCount)
	weightShape := mustShapeForTest(testingObject, []int{outCount, inCount})
	stateShape := mustShapeForTest(testingObject, []int{inCount})
	outShape := mustShapeForTest(testingObject, []int{outCount})
	weights := uploadDTypeTensorForTest(testingObject, backend, weightShape, storageDType, fixture.weightBytes)
	state := uploadDTypeTensorForTest(testingObject, backend, stateShape, storageDType, fixture.stateBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(weights, state, out)

	err := lookupResearchBinaryKernel(testingObject, "pc_prediction", storageDType).Run(weights, state, out)
	convey.So(err, convey.ShouldBeNil)
	assertResearchBytesForTest(
		testingObject, backend, out, storageDType, fixture.predictionValues, researchAccumULP(storageDType),
	)
}

func runPCPredictionErrorParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := pcPredictionErrorFixtureForTest(storageDType, elementCount)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	observed := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.observedBytes)
	predicted := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.predictedBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	defer closeBenchmarkTensors(observed, predicted, out)

	err := lookupResearchBinaryKernel(testingObject, "pc_prediction_error", storageDType).Run(
		observed, predicted, out,
	)
	convey.So(err, convey.ShouldBeNil)
	assertResearchBytesForTest(testingObject, backend, out, storageDType, fixture.expectedValues, 2)
}

func runPCUpdateRepresentationParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	outCount int,
	inCount int,
) {
	fixture := pcFixtureForTest(storageDType, outCount, inCount)
	weights, state, predictionError, out := pcUpdateTensorsForTest(
		testingObject, backend, storageDType, fixture, outCount, inCount, false,
	)
	defer closeBenchmarkTensors(weights, state, predictionError, out)

	err := lookupResearchTernaryKernel(testingObject, "pc_update_representation", storageDType).Run(
		weights, state, predictionError, out,
	)
	convey.So(err, convey.ShouldBeNil)
	assertResearchBytesForTest(
		testingObject,
		backend,
		out,
		storageDType,
		fixture.updatedRepresentationValues,
		researchAccumULP(storageDType),
	)
}

func runPCUpdateWeightsParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	outCount int,
	inCount int,
) {
	fixture := pcFixtureForTest(storageDType, outCount, inCount)
	weights, state, predictionError, out := pcUpdateTensorsForTest(
		testingObject, backend, storageDType, fixture, outCount, inCount, true,
	)
	defer closeBenchmarkTensors(weights, state, predictionError, out)

	err := lookupResearchTernaryKernel(testingObject, "pc_update_weights", storageDType).Run(
		weights, state, predictionError, out,
	)
	convey.So(err, convey.ShouldBeNil)
	assertResearchBytesForTest(
		testingObject, backend, out, storageDType, fixture.updatedWeightValues, 2,
	)
}

func pcUpdateTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	fixture pcFixture,
	outCount int,
	inCount int,
	weightOutput bool,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	weightShape := mustShapeForTest(testingObject, []int{outCount, inCount})
	stateShape := mustShapeForTest(testingObject, []int{inCount})
	errorShape := mustShapeForTest(testingObject, []int{outCount})
	outShape := stateShape

	if weightOutput {
		outShape = weightShape
	}

	weights := uploadDTypeTensorForTest(testingObject, backend, weightShape, storageDType, fixture.weightBytes)
	state := uploadDTypeTensorForTest(testingObject, backend, stateShape, storageDType, fixture.stateBytes)
	predictionError := uploadDTypeTensorForTest(
		testingObject, backend, errorShape, storageDType, fixture.errorBytes,
	)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	return weights, state, predictionError, out
}

func lookupResearchBinaryKernel(
	testingObject testing.TB,
	name string,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation(name, kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType, storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s %s kernel", storageDType.Name(), name)
	}

	return kernel
}

func lookupResearchUnaryKernel(
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

func lookupResearchTernaryKernel(
	testingObject testing.TB,
	name string,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation(name, kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType,
			storageDType,
			storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s %s kernel", storageDType.Name(), name)
	}

	return kernel
}

func assertResearchBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	storageDType dtype.DType,
	expectedValues []float32,
	maxULP uint32,
) {
	testingObject.Helper()

	if storageDType == dtype.Float32 {
		assertFloat32TensorForTest(testingObject, backend, input, expectedValues, maxULP)
		return
	}

	expectedBytes := encodeResearchValuesAsDType(expectedValues, storageDType)
	assertDTypeBytesForTest(testingObject, backend, input, storageDType, expectedBytes, maxULP)
}

func researchAccumULP(storageDType dtype.DType) uint32 {
	if storageDType == dtype.Float32 {
		return 16
	}

	return 2
}

func encodeResearchValuesAsDType(values []float32, storageDType dtype.DType) []byte {
	if storageDType == dtype.Float32 {
		return dtypeconvert.Float32ToBytes(values)
	}

	return encodeFloat32ValuesAsDType(values, storageDType)
}
