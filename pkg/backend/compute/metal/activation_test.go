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

			values, err := output.CloneFloat64()
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
