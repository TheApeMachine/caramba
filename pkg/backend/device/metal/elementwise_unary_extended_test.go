package metal

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	cpumath "github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

type extendedUnaryCase struct {
	name        string
	operation   metalUnaryFloat32Operation
	f32MaxULP   uint32
	dtypeMaxULP uint32
}

var extendedUnaryCases = []extendedUnaryCase{
	{name: "rsqrt", operation: metalUnaryFloat32Rsqrt, f32MaxULP: 2, dtypeMaxULP: 2},
	{name: "exp", operation: metalUnaryFloat32Exp, f32MaxULP: 8, dtypeMaxULP: 2},
	{name: "log", operation: metalUnaryFloat32Log, f32MaxULP: 8, dtypeMaxULP: 2},
	{name: "sin", operation: metalUnaryFloat32Sin, f32MaxULP: 8, dtypeMaxULP: 2},
	{name: "cos", operation: metalUnaryFloat32Cos, f32MaxULP: 8, dtypeMaxULP: 2},
	{name: "tanh", operation: metalUnaryFloat32Tanh, f32MaxULP: 8, dtypeMaxULP: 2},
	{name: "gelu", operation: metalUnaryFloat32Gelu, f32MaxULP: 2, dtypeMaxULP: 2},
	{name: "sigmoid", operation: metalUnaryFloat32Sigmoid, f32MaxULP: 8, dtypeMaxULP: 2},
	{name: "silu", operation: metalUnaryFloat32Silu, f32MaxULP: 8, dtypeMaxULP: 2},
	{name: "swish", operation: metalUnaryFloat32Swish, f32MaxULP: 8, dtypeMaxULP: 2},
	{name: "softsign", operation: metalUnaryFloat32Softsign, f32MaxULP: 2, dtypeMaxULP: 2},
	{name: "elu", operation: metalUnaryFloat32ELU, f32MaxULP: 8, dtypeMaxULP: 2},
	{name: "selu", operation: metalUnaryFloat32SELU, f32MaxULP: 8, dtypeMaxULP: 2},
	{name: "leaky_relu", operation: metalUnaryFloat32LeakyReLU, f32MaxULP: 1, dtypeMaxULP: 1},
	{name: "hardsigmoid", operation: metalUnaryFloat32HardSigmoid, f32MaxULP: 16, dtypeMaxULP: 1},
	{name: "hardswish", operation: metalUnaryFloat32HardSwish, f32MaxULP: 2, dtypeMaxULP: 2},
}

func TestKernelRegistry_MetalExtendedUnaryElementwiseDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalExtendedUnaryDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalExtendedUnaryElementwiseDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalExtendedUnaryElementwiseDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, testCase := range extendedUnaryCases {
		testCase := testCase

		testingObject.Run(testCase.name, func(testingObject *testing.T) {
			for _, elementCount := range parityElementCounts {
				elementCount := elementCount

				testingObject.Run(fmt.Sprintf("N=%d", elementCount), func(testingObject *testing.T) {
					convey.Convey("Given one Metal "+storageDType.Name()+" tensor for "+testCase.name, testingObject, func() {
						assertMetalExtendedUnaryElementwise(testingObject, backend, storageDType, testCase, elementCount)
					})
				})
			}
		})
	}
}

func assertMetalExtendedUnaryElementwise(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	testCase extendedUnaryCase,
	elementCount int,
) {
	testingObject.Helper()

	kernel := lookupUnaryElementwiseKernel(testingObject, testCase.name, storageDType)
	shape, err := tensor.NewShape([]int{elementCount})
	convey.So(err, convey.ShouldBeNil)

	inputBytes, expectedBytes := extendedUnaryBytes(elementCount, testCase.name, storageDType)
	input := uploadTensorBytesForTest(testingObject, backend, shape, storageDType, inputBytes)
	defer func() {
		convey.So(input.Close(), convey.ShouldBeNil)
	}()

	out, err := backend.bridge.empty(shape, storageDType)
	convey.So(err, convey.ShouldBeNil)
	defer func() {
		convey.So(out.Close(), convey.ShouldBeNil)
	}()

	err = kernel.Run(input, out)
	convey.So(err, convey.ShouldBeNil)
	assertExtendedUnaryOutput(testingObject, backend, out, storageDType, expectedBytes, testCase)
}

func extendedUnaryBytes(
	elementCount int,
	name string,
	storageDType dtype.DType,
) ([]byte, []byte) {
	inputValues := extendedUnaryInputValues(elementCount, name)

	if storageDType == dtype.Float32 {
		return dtypeconvert.Float32ToBytes(inputValues),
			dtypeconvert.Float32ToBytes(extendedUnaryExpectedValues(inputValues, name))
	}

	inputBytes := encodeFloat32ValuesAsDType(inputValues, storageDType)
	storedValues := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	expectedValues := extendedUnaryExpectedValues(storedValues, name)

	return inputBytes, encodeFloat32ValuesAsDType(expectedValues, storageDType)
}

func uploadTensorBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	shape tensor.Shape,
	storageDType dtype.DType,
	bytes []byte,
) tensor.Tensor {
	testingObject.Helper()

	input, err := backend.Upload(shape, storageDType, bytes)
	if err != nil {
		testingObject.Fatal(err)
	}

	return input
}

func assertExtendedUnaryOutput(
	testingObject testing.TB,
	backend *Backend,
	out tensor.Tensor,
	storageDType dtype.DType,
	expectedBytes []byte,
	testCase extendedUnaryCase,
) {
	testingObject.Helper()

	actualDType, actualBytes, err := backend.Download(out)
	if err != nil {
		testingObject.Fatalf("Download failed: %v", err)
	}

	if actualDType != storageDType {
		testingObject.Fatalf("download dtype mismatch: got %s want %s", actualDType, storageDType)
	}

	if storageDType == dtype.Float32 {
		assertFloat32WithinULP(
			testingObject,
			mustFloat32Bytes(actualBytes),
			mustFloat32Bytes(expectedBytes),
			testCase.f32MaxULP,
		)
		return
	}

	assertDTypeBytesWithinULP(testingObject, actualBytes, expectedBytes, testCase.dtypeMaxULP)
}

func mustFloat32Bytes(bytes []byte) []float32 {
	values, err := dtypeconvert.BytesToFloat32(dtype.Float32, bytes)
	if err != nil {
		panic(err)
	}

	return values
}

func extendedUnaryInputValues(elementCount int, name string) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = extendedUnaryInputValue(index, name)
	}

	return values
}

func extendedUnaryInputValue(index int, name string) float32 {
	if name == "log" || name == "rsqrt" {
		return 0.25 + float32(index%251)/16
	}

	if name == "exp" {
		return float32(index%161)/16 - 5
	}

	if name == "gelu" {
		return float32(1+index%240)/12 - 4
	}

	return float32(index%257)/16 - 8
}

func extendedUnaryExpectedValues(input []float32, name string) []float32 {
	out := make([]float32, len(input))

	for index, value := range input {
		out[index] = extendedUnaryExpected(value, name)
	}

	return out
}

func extendedUnaryExpected(value float32, name string) float32 {
	switch name {
	case "rsqrt":
		return 1 / float32(math.Sqrt(float64(value)))
	case "exp":
		return float32(math.Exp(float64(value)))
	case "log":
		return float32(math.Log(float64(value)))
	case "sin":
		return float32(math.Sin(float64(value)))
	case "cos":
		return float32(math.Cos(float64(value)))
	case "tanh":
		return float32(math.Tanh(float64(value)))
	case "gelu":
		return cpumath.FastGelu32(value)
	case "sigmoid":
		return 1 / (1 + float32(math.Exp(float64(-value))))
	case "silu", "swish":
		return value / (1 + float32(math.Exp(float64(-value))))
	case "softsign":
		return value / (1 + float32(math.Abs(float64(value))))
	case "elu":
		return extendedELUExpected(value)
	case "selu":
		return extendedSELUExpected(value)
	case "leaky_relu":
		return extendedLeakyReLUExpected(value)
	case "hardsigmoid":
		return min(max(value/6+0.5, 0), 1)
	case "hardswish":
		return value * min(max((value+3)/6, 0), 1)
	}

	panic("unknown extended unary operation: " + name)
}

func extendedELUExpected(value float32) float32 {
	if value > 0 {
		return value
	}

	return float32(math.Exp(float64(value))) - 1
}

func extendedSELUExpected(value float32) float32 {
	const alpha = float32(1.6732632423543772)
	const scale = float32(1.0507009873554805)

	if value > 0 {
		return scale * value
	}

	return scale * alpha * (float32(math.Exp(float64(value))) - 1)
}

func extendedLeakyReLUExpected(value float32) float32 {
	if value > 0 {
		return value
	}

	return 0.01 * value
}
