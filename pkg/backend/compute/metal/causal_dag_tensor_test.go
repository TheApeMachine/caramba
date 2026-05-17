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

func TestMetalCausalOps_DAGMarkovFactorizationTensor(test *testing.T) {
	Convey("Given resident Metal causal DAG Markov inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		causalOps := causalOpsForTest(test, tensorBackend)

		Convey("It should match the scalar DAG Markov reference at contract sizes", func() {
			for _, samples := range metalContractSizes() {
				nodeCount, observations, adjacency := metalCausalDAGInputs(samples)
				expected := referenceCausalDAG(test, nodeCount, samples, observations, adjacency)
				output, err := causalOps.DAGMarkovFactorizationTensor(
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, samples, nodeCount), observations),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, nodeCount, nodeCount), adjacency),
					causalShape(test, samples),
					nodeCount,
					samples,
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, expected, 2e-2)
			}
		})
	})
}

func TestTensorBackend_applyCausalDAGMarkovFactorizationGraph(test *testing.T) {
	Convey("Given Metal causal DAG Markov graph execution", test, func() {
		Convey("It should keep DAG Markov graph outputs resident", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			graph, target, expectedBytes, expected := causalDAGGraph(test)

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

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			assertMetalMaxDiff(values, expected, 2e-2)
		})
	})
}

func BenchmarkMetalCausalOps_DAGMarkovFactorizationTensor(benchmark *testing.B) {
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

	samples := 8192
	nodeCount, observations, adjacency := metalCausalDAGInputs(samples)
	observationTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, samples, nodeCount), observations)
	adjacencyTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, nodeCount, nodeCount), adjacency)
	defer closeBenchmarkTensors(observationTensor, adjacencyTensor)

	outputShape := causalShape(benchmark, samples)
	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := causalOps.DAGMarkovFactorizationTensor(
			observationTensor,
			adjacencyTensor,
			outputShape,
			nodeCount,
			samples,
		)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func referenceCausalDAG(
	test testing.TB,
	nodeCount int,
	samples int,
	observations []float64,
	adjacency []float64,
) []float64 {
	test.Helper()

	cpuState, err := cpucausal.NewDAGMarkovFactorization().Forward(
		state.NewDict().WithShape([]int{nodeCount, samples}).WithInputs(
			observations,
			adjacency,
		),
	)
	So(err, ShouldBeNil)

	return cpuState.Out
}

func causalDAGGraph(test testing.TB) (*ir.Graph, *ir.Node, int64, []float64) {
	test.Helper()

	samples := 64
	nodeCount, observations, adjacency := metalCausalDAGInputs(samples)
	values := [][]float64{observations, adjacency}
	names := []string{"dag_observations", "dag_adjacency"}
	graph := ir.NewGraph()
	inputs := make([]*ir.Node, len(values))
	expectedBytes := int64(0)

	for index, value := range values {
		node := ir.NewNode(names[index], ir.OpInput, causalShape(test, len(value)))
		node.SetMetadata("values", value)
		graph.AddNode(node)
		inputs[index] = node
		expectedBytes += int64(len(value) * 4)
	}

	target := ir.NewNode(
		"causal_dag_markov",
		"causal.dag_markov_factorization",
		causalShape(test, nodeCount, samples),
	)
	for _, input := range inputs {
		target.AddInput(input)
	}
	graph.AddNode(target)

	return graph, target, expectedBytes, referenceCausalDAG(
		test,
		nodeCount,
		samples,
		observations,
		adjacency,
	)
}
