//go:build darwin && cgo

package metal

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	cpuactivation "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/activation"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestMetalActivation_GELU(test *testing.T) {
	lib := metallibPathOrSkip(test, "activation.metallib")

	Convey("Given Metal GELU activation", test, func() {
		activationOps, err := New(lib)
		So(err, ShouldBeNil)

		input := []float64{-12, -6, -5, -3, -1, 0, 1, 3, 5, 6, 12}
		expectedState, err := cpuactivation.NewGelu().Forward(
			state.NewDict().WithShape([]int{len(input)}).WithInput(input),
		)
		So(err, ShouldBeNil)

		Convey("It should match the CPU kernel for bounded and saturated values", func() {
			output, err := activationOps.GELU(input)

			So(err, ShouldBeNil)
			assertAlmostEqualSlice(output, expectedState.Out, 1e-4)
		})

		Convey("It should match the CPU kernel for resident tensors", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			shape, err := computetensor.NewShape([]int{len(input)})
			So(err, ShouldBeNil)

			output, err := activationOps.GELUTensor(
				uploadMetalTensorForTest(test, tensorBackend, shape, input),
			)

			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := tensorFloat64Values(output)
			So(err, ShouldBeNil)
			assertAlmostEqualSlice(values, expectedState.Out, 1e-4)
		})
	})
}

func TestMetalActivation_SwiGLU(test *testing.T) {
	lib := metallibPathOrSkip(test, "activation.metallib")

	Convey("Given Metal SwiGLU activation", test, func() {
		activationOps, err := New(lib)
		So(err, ShouldBeNil)

		input := []float64{
			-3, -1.5, -0.25, 0, 0.75, 2, 4,
			0.5, -1.25, 3, 7, -0.5, 1.75, -2,
		}
		expectedState, err := cpuactivation.NewSwiGLU().Forward(
			state.NewDict().WithShape([]int{len(input)}).WithInput(input),
		)
		So(err, ShouldBeNil)

		Convey("It should match the CPU gate*sigmoid(gate)*value contract", func() {
			output, err := activationOps.SwiGLU(input)

			So(err, ShouldBeNil)
			assertAlmostEqualSlice(output, expectedState.Out, 1e-5)
		})

		Convey("It should match the CPU contract for resident tensors", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			shape, err := computetensor.NewShape([]int{len(input)})
			So(err, ShouldBeNil)

			output, err := activationOps.SwiGLUTensor(
				uploadMetalTensorForTest(test, tensorBackend, shape, input),
			)

			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := tensorFloat64Values(output)
			So(err, ShouldBeNil)
			assertAlmostEqualSlice(values, expectedState.Out, 1e-5)
		})

		Convey("It should split gates and values inside each resident tensor row", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			input := []float64{
				1, 2, 3, 10, 20, 30,
				4, 5, 6, 40, 50, 60,
			}
			shape, err := computetensor.NewShape([]int{2, 6})
			So(err, ShouldBeNil)
			expectedState, err := cpuactivation.NewSwiGLU().Forward(
				state.NewDict().WithShape([]int{2, 6}).WithInput(input),
			)
			So(err, ShouldBeNil)

			output, err := activationOps.SwiGLUTensor(
				uploadMetalTensorForTest(test, tensorBackend, shape, input),
			)

			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := tensorFloat64Values(output)
			So(err, ShouldBeNil)
			assertAlmostEqualSlice(values, expectedState.Out, 1e-4)
		})
	})
}

func BenchmarkMetalActivation_GELUTensor(benchmark *testing.B) {
	lib := metallibPathOrSkip(benchmark, "activation.metallib")
	tensorBackend, err := NewTensorBackend()

	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}

	defer func() {
		_ = tensorBackend.Close()
	}()

	activationOps, err := New(lib)

	if err != nil {
		benchmark.Fatal(err)
	}

	shape, err := computetensor.NewShape([]int{4096})

	if err != nil {
		benchmark.Fatal(err)
	}

	input := uploadMetalTensor(tensorBackend, shape, make([]float64, shape.Len()))
	defer func() {
		_ = input.Close()
	}()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		output, err := activationOps.GELUTensor(input)

		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkMetalActivation_SwiGLUTensor(benchmark *testing.B) {
	lib := metallibPathOrSkip(benchmark, "activation.metallib")
	tensorBackend, err := NewTensorBackend()

	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}

	defer func() {
		_ = tensorBackend.Close()
	}()

	activationOps, err := New(lib)

	if err != nil {
		benchmark.Fatal(err)
	}

	shape, err := computetensor.NewShape([]int{1, 8192})

	if err != nil {
		benchmark.Fatal(err)
	}

	inputValues := make([]float64, shape.Len())

	for index := range inputValues {
		inputValues[index] = float64(index%257)/64 - 2
	}

	input := uploadMetalTensor(tensorBackend, shape, inputValues)
	defer func() {
		_ = input.Close()
	}()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		output, err := activationOps.SwiGLUTensor(input)

		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}
