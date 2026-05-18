//go:build darwin && cgo

package metal

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	cpupositional "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/positional"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestMetalPositional_ALiBiTensor(test *testing.T) {
	Convey("Given resident Metal ALiBi output shapes", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		positionalOps := positionalOpsForTest(test, tensorBackend)

		Convey("It should match the CPU ALiBi contract", func() {
			for _, causal := range []bool{false, true} {
				shape, err := computetensor.NewShape([]int{3, 4, 5})
				So(err, ShouldBeNil)

				output, err := positionalOps.ALiBiTensor(shape, causal)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := tensorFloat64Values(output)
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, referenceALiBi(shape.Dims(), causal), 1e-6)
			}
		})
	})
}

func TestTensorBackend_applyALiBiGraph(test *testing.T) {
	Convey("Given Metal ALiBi graph execution", test, func() {
		Convey("It should execute without input readback", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			shape, err := computetensor.NewShape([]int{2, 3, 4})
			So(err, ShouldBeNil)

			output := ir.NewNode("alibi", "positional.alibi", shape)
			output.SetMetadata("causal", true)
			graph := ir.NewGraph()
			graph.AddNode(output)

			before := tensorBackend.runtime.Metrics()
			results, err := NewRunnerWithBackend(tensorBackend).Execute(
				context.Background(),
				graph,
				[]*ir.Node{output},
			)
			after := tensorBackend.runtime.Metrics()
			So(err, ShouldBeNil)
			So(results, ShouldHaveLength, 1)
			So(results["alibi"].Location(), ShouldEqual, computetensor.Metal)
			So(after.TransferBytes-before.TransferBytes, ShouldEqual, int64(0))

			values, err := tensorFloat64Values(results["alibi"])
			So(err, ShouldBeNil)
			defer func() {
				So(results["alibi"].Close(), ShouldBeNil)
			}()
			assertMetalMaxDiff(values, referenceALiBi(shape.Dims(), true), 1e-6)
		})
	})
}

func BenchmarkMetalPositional_ALiBiTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()

	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}
	defer func() {
		_ = tensorBackend.Close()
	}()

	positionalOps, err := tensorBackend.positional()
	if err != nil {
		benchmark.Fatal(err)
	}

	shape, err := computetensor.NewShape([]int{16, 128, 128})
	if err != nil {
		benchmark.Fatal(err)
	}

	benchmark.ResetTimer()

	for benchmark.Loop() {
		output, err := positionalOps.ALiBiTensor(shape, true)
		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func positionalOpsForTest(test testing.TB, tensorBackend *TensorBackend) *MetalPositional {
	test.Helper()

	positionalOps, err := tensorBackend.positional()
	So(err, ShouldBeNil)

	return positionalOps
}

func referenceALiBi(shape []int, causal bool) []float64 {
	stateDict := state.NewDict().WithShape(shape)
	stateDict.Causal = causal

	expected, err := cpupositional.NewALiBi().Forward(stateDict)
	So(err, ShouldBeNil)

	return expected.Out
}
