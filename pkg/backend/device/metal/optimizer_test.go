package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

const optimizerMaxULP = uint32(8)

type optimizer4Case struct {
	name     string
	maxULP   uint32
	expected func(
		params []float32,
		gradients []float32,
		first []float32,
		second []float32,
	) []float32
}

type optimizer3Case struct {
	name     string
	maxULP   uint32
	expected func(params []float32, gradients []float32, state []float32) []float32
}

var optimizer4Cases = []optimizer4Case{
	{name: "adam_step", maxULP: optimizerMaxULP, expected: optimizerAdamExpected},
	{name: "adamw_step", maxULP: optimizerMaxULP, expected: optimizerAdamWExpected},
	{name: "adamax_step", maxULP: 32, expected: optimizerAdamaxExpected},
}

var optimizer3Cases = []optimizer3Case{
	{name: "adagrad_step", maxULP: optimizerMaxULP, expected: optimizerAdagradExpected},
	{name: "rmsprop_step", maxULP: optimizerMaxULP, expected: optimizerRMSpropExpected},
	{name: "lion_step", maxULP: optimizerMaxULP, expected: optimizerLionExpected},
	{name: "sgd_step", maxULP: optimizerMaxULP, expected: optimizerSGDExpected},
	{name: "lars_step", maxULP: optimizerMaxULP, expected: optimizerLARSExpected},
}

func TestKernelRegistry_MetalOptimizerDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalOptimizerDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalOptimizerDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalOptimizerDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			for _, testCase := range optimizer4Cases {
				testCase := testCase

				testingObject.Run(testCase.name, func(testingObject *testing.T) {
					convey.Convey("Given Metal "+storageDType.Name()+" optimizer tensors", testingObject, func() {
						runOptimizer4ParityCase(testingObject, backend, storageDType, elementCount, testCase)
					})
				})
			}

			for _, testCase := range optimizer3Cases {
				testCase := testCase

				testingObject.Run(testCase.name, func(testingObject *testing.T) {
					convey.Convey("Given Metal "+storageDType.Name()+" optimizer tensors", testingObject, func() {
						runOptimizer3ParityCase(testingObject, backend, storageDType, elementCount, testCase)
					})
				})
			}

			testingObject.Run("lbfgs_step", func(testingObject *testing.T) {
				convey.Convey("Given Metal "+storageDType.Name()+" optimizer tensors", testingObject, func() {
					runOptimizer2ParityCase(testingObject, backend, storageDType, elementCount)
				})
			})

			testingObject.Run("hebbian_step", func(testingObject *testing.T) {
				convey.Convey("Given Metal "+storageDType.Name()+" optimizer tensors", testingObject, func() {
					runHebbianParityCase(testingObject, backend, storageDType, elementCount)
				})
			})
		})
	}
}

func runOptimizer4ParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
	testCase optimizer4Case,
) {
	defer configureOptimizerExpectedArithmetic(storageDType)()

	paramBytes, gradientBytes, params, gradients :=
		optimizerStorageInputs(elementCount, storageDType)
	firstInitial := optimizerStateValues(elementCount, 3)
	secondInitial := optimizerStateValues(elementCount, 5)
	expectedFirst := append([]float32(nil), firstInitial...)
	expectedSecond := append([]float32(nil), secondInitial...)
	expectedOut := testCase.expected(params, gradients, expectedFirst, expectedSecond)
	tensors := optimizer4TensorsForTest(
		testingObject, backend, storageDType, elementCount,
		paramBytes, gradientBytes, firstInitial, secondInitial,
	)
	defer closeBenchmarkTensors(tensors...)

	err := lookupOptimizer4Kernel(testingObject, testCase.name, storageDType).Run(tensors...)
	convey.So(err, convey.ShouldBeNil)
	assertOptimizerStorageForTest(
		testingObject, backend, tensors[4], storageDType, expectedOut, testCase.maxULP,
	)
	assertOptimizerStateForTest(testingObject, backend, tensors[2], expectedFirst)
	assertOptimizerStateForTest(testingObject, backend, tensors[3], expectedSecond)
}

func runOptimizer3ParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
	testCase optimizer3Case,
) {
	defer configureOptimizerExpectedArithmetic(storageDType)()

	paramBytes, gradientBytes, params, gradients :=
		optimizerStorageInputs(elementCount, storageDType)
	stateInitial := optimizerStateValues(elementCount, 7)
	expectedState := append([]float32(nil), stateInitial...)
	expectedOut := testCase.expected(params, gradients, expectedState)
	tensors := optimizer3TensorsForTest(
		testingObject, backend, storageDType, elementCount,
		paramBytes, gradientBytes, stateInitial,
	)
	defer closeBenchmarkTensors(tensors...)

	err := lookupOptimizer3Kernel(testingObject, testCase.name, storageDType).Run(tensors...)
	convey.So(err, convey.ShouldBeNil)
	assertOptimizerStorageForTest(
		testingObject, backend, tensors[3], storageDType, expectedOut, testCase.maxULP,
	)
	assertOptimizerStateForTest(testingObject, backend, tensors[2], expectedState)
}

func runOptimizer2ParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	defer configureOptimizerExpectedArithmetic(storageDType)()

	paramBytes, gradientBytes, params, gradients :=
		optimizerStorageInputs(elementCount, storageDType)
	expectedOut := optimizerLBFGSExpected(params, gradients)
	tensors := optimizer2TensorsForTest(
		testingObject, backend, storageDType, elementCount, paramBytes, gradientBytes,
	)
	defer closeBenchmarkTensors(tensors...)

	err := lookupOptimizer2Kernel(testingObject, "lbfgs_step", storageDType).Run(tensors...)
	convey.So(err, convey.ShouldBeNil)
	assertOptimizerStorageForTest(
		testingObject, backend, tensors[2], storageDType, expectedOut, optimizerMaxULP,
	)
}

func runHebbianParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	defer configureOptimizerExpectedArithmetic(storageDType)()

	weightBytes, postBytes, preBytes, expectedOut :=
		hebbianDTypeBytes(elementCount, storageDType)
	weights, post, pre, out := hebbianTensorsForTest(
		testingObject, backend, storageDType, elementCount, weightBytes, postBytes, preBytes,
	)
	defer closeBenchmarkTensors(weights, post, pre, out)

	err := lookupHebbianKernel(testingObject, storageDType).Run(weights, post, pre, out)
	convey.So(err, convey.ShouldBeNil)
	assertOptimizerStorageForTest(
		testingObject, backend, out, storageDType, expectedOut, optimizerMaxULP,
	)
}

func optimizer4TensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
	paramBytes []byte,
	gradientBytes []byte,
	firstState []float32,
	secondState []float32,
) []tensor.Tensor {
	shape := mustShapeForTest(testingObject, []int{elementCount})
	params := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, paramBytes)
	gradients := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, gradientBytes)
	first := uploadDTypeTensorForTest(
		testingObject, backend, shape, dtype.Float32, optimizerStateBytes(firstState),
	)
	second := uploadDTypeTensorForTest(
		testingObject, backend, shape, dtype.Float32, optimizerStateBytes(secondState),
	)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)

	return []tensor.Tensor{params, gradients, first, second, out}
}

func optimizer3TensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
	paramBytes []byte,
	gradientBytes []byte,
	stateValues []float32,
) []tensor.Tensor {
	shape := mustShapeForTest(testingObject, []int{elementCount})
	params := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, paramBytes)
	gradients := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, gradientBytes)
	state := uploadDTypeTensorForTest(
		testingObject, backend, shape, dtype.Float32, optimizerStateBytes(stateValues),
	)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)

	return []tensor.Tensor{params, gradients, state, out}
}
