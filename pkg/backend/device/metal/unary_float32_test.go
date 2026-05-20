package metal

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

type unaryFloat32Case struct {
	name      string
	operation metalUnaryFloat32Operation
	apply     func(*Backend, context.Context, tensor.Tensor) (tensor.Tensor, error)
	maxULP    uint32
}

var unaryFloat32Cases = []unaryFloat32Case{
	{name: "relu", operation: metalUnaryFloat32Relu, apply: (*Backend).ReluFloat32},
	{name: "abs", operation: metalUnaryFloat32Abs, apply: (*Backend).AbsFloat32},
	{name: "neg", operation: metalUnaryFloat32Neg, apply: (*Backend).NegFloat32},
	{name: "square", operation: metalUnaryFloat32Square, apply: (*Backend).SquareFloat32},
	{name: "recip", operation: metalUnaryFloat32Recip, apply: (*Backend).RecipFloat32},
	{name: "sqrt", operation: metalUnaryFloat32Sqrt, apply: (*Backend).SqrtFloat32, maxULP: 1},
	{name: "sign", operation: metalUnaryFloat32Sign, apply: (*Backend).SignFloat32},
}

func TestBackend_ReluFloat32(t *testing.T) {
	testBackendUnaryFloat32(t, unaryFloat32Cases[0])
}

func TestBackend_AbsFloat32(t *testing.T) {
	testBackendUnaryFloat32(t, unaryFloat32Cases[1])
}

func TestBackend_NegFloat32(t *testing.T) {
	testBackendUnaryFloat32(t, unaryFloat32Cases[2])
}

func TestBackend_SquareFloat32(t *testing.T) {
	testBackendUnaryFloat32(t, unaryFloat32Cases[3])
}

func TestBackend_RecipFloat32(t *testing.T) {
	testBackendUnaryFloat32(t, unaryFloat32Cases[4])
}

func TestBackend_SqrtFloat32(t *testing.T) {
	testBackendUnaryFloat32(t, unaryFloat32Cases[5])
}

func TestBackend_SignFloat32(t *testing.T) {
	testBackendUnaryFloat32(t, unaryFloat32Cases[6])
}

func testBackendUnaryFloat32(t *testing.T, testCase unaryFloat32Case) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		t.Run(fmt.Sprintf("N=%d", elementCount), func(t *testing.T) {
			convey.Convey("Given one Metal float32 tensor for "+testCase.name, t, func() {
				shape, err := tensor.NewShape([]int{elementCount})
				convey.So(err, convey.ShouldBeNil)

				inputValues, expectedValues := unaryFloat32ParityValues(
					elementCount,
					testCase.name,
				)

				input, err := backend.Upload(shape, dtype.Float32, dtypeconvert.Float32ToBytes(inputValues))
				convey.So(err, convey.ShouldBeNil)
				defer func() {
					convey.So(input.Close(), convey.ShouldBeNil)
				}()

				out, err := testCase.apply(backend, context.Background(), input)
				convey.So(err, convey.ShouldBeNil)
				defer func() {
					convey.So(out.Close(), convey.ShouldBeNil)
				}()

				actual := downloadFloat32ForTest(t, backend, out)
				convey.So(len(actual), convey.ShouldEqual, elementCount)
				assertUnaryFloat32Parity(t, actual, expectedValues, testCase.maxULP)
			})
		})
	}
}

func TestKernelRegistry_MetalUnaryFloat32(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	for _, testCase := range unaryFloat32Cases {
		testCase := testCase

		t.Run(testCase.name, func(t *testing.T) {
			convey.Convey("Given the device kernel registry for "+testCase.name, t, func() {
				kernel, ok := kernels.Default.LookupLocation(testCase.name, kernels.Signature{
					Layout:  tensor.LayoutDense,
					Inputs:  []dtype.DType{dtype.Float32},
					Outputs: []dtype.DType{dtype.Float32},
				}, tensor.Metal)
				convey.So(ok, convey.ShouldBeTrue)

				shape, err := tensor.NewShape([]int{1})
				convey.So(err, convey.ShouldBeNil)

				inputValues, expectedValues := unaryFloat32ParityValues(1, testCase.name)
				input, err := backend.Upload(
					shape,
					dtype.Float32,
					dtypeconvert.Float32ToBytes(inputValues),
				)
				convey.So(err, convey.ShouldBeNil)
				defer func() {
					convey.So(input.Close(), convey.ShouldBeNil)
				}()

				out, err := backend.bridge.empty(shape, dtype.Float32)
				convey.So(err, convey.ShouldBeNil)
				defer func() {
					convey.So(out.Close(), convey.ShouldBeNil)
				}()

				err = kernel.Run(input, out)
				convey.So(err, convey.ShouldBeNil)
				assertUnaryFloat32Parity(
					t,
					downloadFloat32ForTest(t, backend, out),
					expectedValues,
					testCase.maxULP,
				)
			})
		})
	}
}

func BenchmarkBackend_UnaryFloat32(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, testCase := range unaryFloat32Cases {
		testCase := testCase

		benchmark.Run(testCase.name, func(benchmark *testing.B) {
			for _, elementCount := range parityElementCounts {
				elementCount := elementCount

				benchmark.Run(fmt.Sprintf("N=%d", elementCount), func(benchmark *testing.B) {
					benchmarkBackendUnaryFloat32(benchmark, backend, testCase, elementCount)
				})
			}
		})
	}
}

func BenchmarkKernel_RunUnaryFloat32(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, testCase := range unaryFloat32Cases {
		testCase := testCase

		benchmark.Run(testCase.name, func(benchmark *testing.B) {
			for _, elementCount := range parityElementCounts {
				elementCount := elementCount

				benchmark.Run(fmt.Sprintf("N=%d", elementCount), func(benchmark *testing.B) {
					benchmarkKernelRunUnaryFloat32(benchmark, backend, testCase, elementCount)
				})
			}
		})
	}
}

func benchmarkBackendUnaryFloat32(
	benchmark *testing.B,
	backend *Backend,
	testCase unaryFloat32Case,
	elementCount int,
) {
	benchmark.Helper()

	shape, input := uploadUnaryFloat32BenchmarkInput(
		benchmark,
		backend,
		testCase.name,
		elementCount,
	)
	defer func() {
		_ = input.Close()
	}()

	benchmark.SetBytes(int64(shape.Len() * 2 * 4))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		out, err := testCase.apply(backend, context.Background(), input)
		if err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func benchmarkKernelRunUnaryFloat32(
	benchmark *testing.B,
	backend *Backend,
	testCase unaryFloat32Case,
	elementCount int,
) {
	benchmark.Helper()

	shape, input := uploadUnaryFloat32BenchmarkInput(
		benchmark,
		backend,
		testCase.name,
		elementCount,
	)
	defer func() {
		_ = input.Close()
	}()

	out, err := backend.bridge.empty(shape, dtype.Float32)
	if err != nil {
		benchmark.Fatal(err)
	}
	defer func() {
		_ = out.Close()
	}()

	benchmark.SetBytes(int64(shape.Len() * 2 * 4))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := runMetalUnaryFloat32(testCase.operation, input, out); err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func uploadUnaryFloat32BenchmarkInput(
	testingObject testing.TB,
	backend *Backend,
	name string,
	elementCount int,
) (tensor.Shape, tensor.Tensor) {
	testingObject.Helper()

	shape, err := tensor.NewShape([]int{elementCount})
	if err != nil {
		testingObject.Fatal(err)
	}

	inputValues, _ := unaryFloat32ParityValues(elementCount, name)
	input, err := backend.Upload(shape, dtype.Float32, dtypeconvert.Float32ToBytes(inputValues))
	if err != nil {
		testingObject.Fatal(err)
	}

	return shape, input
}

func unaryFloat32ParityValues(elementCount int, name string) ([]float32, []float32) {
	inputValues := make([]float32, elementCount)
	expectedValues := make([]float32, elementCount)

	for index := range inputValues {
		inputValues[index] = unaryFloat32InputValue(index, name)
		expectedValues[index] = unaryFloat32Expected(name, inputValues[index])
	}

	return inputValues, expectedValues
}

func unaryFloat32InputValue(index int, name string) float32 {
	if name == "sqrt" {
		value := float32(index % 32)

		return value * value
	}

	if name == "recip" {
		integerValue := 1 << uint(index%8)
		value := float32(integerValue)
		if index%3 == 0 {
			return -value
		}

		return value
	}

	return float32((index % 63) - 31)
}

func unaryFloat32Expected(name string, value float32) float32 {
	switch name {
	case "relu":
		if value > 0 {
			return value
		}

		return 0
	case "abs":
		if value < 0 {
			return -value
		}

		return value
	case "neg":
		return -value
	case "square":
		return value * value
	case "recip":
		return 1 / value
	case "sqrt":
		return float32(math.Sqrt(float64(value)))
	case "sign":
		if value > 0 {
			return 1
		}

		if value < 0 {
			return -1
		}

		return 0
	}

	panic("unknown unary float32 operation: " + name)
}

func assertUnaryFloat32Parity(
	testingObject testing.TB,
	actualValues []float32,
	expectedValues []float32,
	maxULP uint32,
) {
	testingObject.Helper()

	if maxULP == 0 {
		assertFloat32BitwiseEqual(testingObject, actualValues, expectedValues)
		return
	}

	assertFloat32WithinULP(testingObject, actualValues, expectedValues, maxULP)
}

func assertFloat32WithinULP(
	testingObject testing.TB,
	actualValues []float32,
	expectedValues []float32,
	maxULP uint32,
) {
	testingObject.Helper()

	if len(actualValues) != len(expectedValues) {
		testingObject.Fatalf("length mismatch: got %d want %d", len(actualValues), len(expectedValues))
	}

	for index := range actualValues {
		distance := float32ULPDistance(actualValues[index], expectedValues[index])
		if distance <= maxULP {
			continue
		}

		testingObject.Fatalf(
			"float32 ULP mismatch at %d: got %08x (%g), want %08x (%g), distance %d > %d",
			index,
			math.Float32bits(actualValues[index]),
			actualValues[index],
			math.Float32bits(expectedValues[index]),
			expectedValues[index],
			distance,
			maxULP,
		)
	}
}

func float32ULPDistance(actual float32, expected float32) uint32 {
	actualBits := orderedFloat32Bits(actual)
	expectedBits := orderedFloat32Bits(expected)

	if actualBits > expectedBits {
		return uint32(actualBits - expectedBits)
	}

	return uint32(expectedBits - actualBits)
}

func orderedFloat32Bits(value float32) int64 {
	bits := int64(int32(math.Float32bits(value)))
	if bits < 0 {
		return int64(-2147483648) - bits
	}

	return bits
}
