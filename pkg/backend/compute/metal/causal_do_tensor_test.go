//go:build darwin && cgo

package metal

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	cpucausal "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/causal"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestMetalCausalOps_DoCalculusTensor(test *testing.T) {
	Convey("Given resident Metal causal do-calculus inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		causalOps := causalOpsForTest(test, tensorBackend)

		Convey("It should match the scalar do-calculus reference at shape sizes", func() {
			for _, nodeCount := range metalDoCalculusSizes() {
				covariance, mask, values := metalCausalDoInputs(nodeCount)
				expected := referenceCausalDo(test, nodeCount, covariance, mask, values)
				output, err := causalOps.DoCalculusTensor(
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, nodeCount, nodeCount), covariance),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, nodeCount), mask),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, nodeCount), values),
					causalShape(test, nodeCount+nodeCount*nodeCount),
					nodeCount,
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				actual, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(actual, expected, 2e-3)
			}
		})
	})
}

func TestTensorBackend_applyCausalDoCalculusGraph(test *testing.T) {
	Convey("Given Metal causal do-calculus graph execution", test, func() {
		Convey("It should keep do-calculus graph outputs resident", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			graph, target, expectedBytes, expected := causalDoGraph(test)

			before := tensorBackend.runtime.Metrics()
			results, err := NewRunnerWithBackend(tensorBackend).Execute(context.Background(), graph, []*ir.Node{target})
			after := tensorBackend.runtime.Metrics()
			So(err, ShouldBeNil)
			So(results, ShouldHaveLength, 1)
			So(after.TransferBytes-before.TransferBytes, ShouldEqual, expectedBytes)

			output := results[target.ID()]
			So(output.Location(), ShouldEqual, computetensor.Metal)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			actual, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			assertMetalMaxDiff(actual, expected, 2e-3)
		})
	})
}

func BenchmarkMetalCausalOps_DoCalculusTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()
	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}
	defer func() {
		_ = tensorBackend.Close()
	}()

	causalOps, err := tensorBackend.causal()
	if err != nil {
		benchmark.Fatal(err)
	}

	nodeCount := 32
	covariance, mask, values := metalCausalDoInputs(nodeCount)
	covarianceTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, nodeCount, nodeCount), covariance)
	maskTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, nodeCount), mask)
	valueTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, nodeCount), values)
	defer closeBenchmarkTensors(covarianceTensor, maskTensor, valueTensor)

	outputShape := causalShape(benchmark, nodeCount+nodeCount*nodeCount)
	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := causalOps.DoCalculusTensor(
			covarianceTensor,
			maskTensor,
			valueTensor,
			outputShape,
			nodeCount,
		)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func metalDoCalculusSizes() []int {
	return []int{1, 3, 7, 16}
}

func metalCausalDoInputs(nodeCount int) ([]float64, []float64, []float64) {
	covariance := make([]float64, nodeCount*nodeCount)
	mask := make([]float64, nodeCount)
	values := make([]float64, nodeCount)

	for row := range nodeCount {
		values[row] = float64(float32(0.15 + 0.05*float64(row%5)))
		for col := range nodeCount {
			entry := 0.01 * float64((row+col)%5+1)
			if row == col {
				entry = 1.5 + 0.03*float64(row%7)
			}

			covariance[row*nodeCount+col] = float64(float32(entry))
			covariance[col*nodeCount+row] = float64(float32(entry))
		}
	}

	for index := range nodeCount {
		if index%3 == 1 || nodeCount == 1 {
			mask[index] = 1
		}
	}

	return covariance, mask, values
}

func referenceCausalDo(
	test testing.TB,
	nodeCount int,
	covariance []float64,
	mask []float64,
	values []float64,
) []float64 {
	test.Helper()

	cpuState, err := cpucausal.NewDoCalculus().Forward(
		state.NewDict().WithShape([]int{nodeCount, nodeCount}).WithInputs(
			covariance,
			mask,
			values,
		),
	)
	So(err, ShouldBeNil)

	return cpuState.Out
}

func causalDoGraph(test testing.TB) (*ir.Graph, *ir.Node, int64, []float64) {
	test.Helper()

	nodeCount := 7
	covariance, mask, values := metalCausalDoInputs(nodeCount)
	inputValues := [][]float64{covariance, mask, values}
	names := []string{"do_covariance", "do_mask", "do_values"}
	graph := ir.NewGraph()
	inputs := make([]*ir.Node, len(inputValues))
	expectedBytes := int64(0)

	for index, value := range inputValues {
		node := ir.NewNode(names[index], ir.OpInput, causalShape(test, len(value)))
		node.SetMetadata("values", value)
		graph.AddNode(node)
		inputs[index] = node
		expectedBytes += int64(len(value) * 4)
	}

	target := ir.NewNode(
		"causal_do_calculus",
		"causal.do_calculus",
		causalShape(test, nodeCount),
	)
	for _, input := range inputs {
		target.AddInput(input)
	}
	graph.AddNode(target)

	return graph, target, expectedBytes, referenceCausalDo(
		test,
		nodeCount,
		covariance,
		mask,
		values,
	)
}
