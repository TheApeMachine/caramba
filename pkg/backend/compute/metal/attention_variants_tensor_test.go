//go:build darwin && cgo

package metal

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	cpuattention "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/attention"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestMetalAttention_MQATensor(test *testing.T) {
	Convey("Given resident Metal MQA inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		attentionOps := attentionOpsForTest(test, tensorBackend)

		Convey("It should match CPU MQA", func() {
			queryShape, keyValueShape := mqaShapes(test)
			query := positionalSequence(queryShape.Len())
			key := positionalSequence(keyValueShape.Len())
			value := positionalSequence(keyValueShape.Len())
			queryTensor := uploadMetalTensorForTest(test, tensorBackend, queryShape, query)
			keyTensor := uploadMetalTensorForTest(test, tensorBackend, keyValueShape, key)
			valueTensor := uploadMetalTensorForTest(test, tensorBackend, keyValueShape, value)

			output, err := attentionOps.MQATensor(
				queryTensor,
				keyTensor,
				valueTensor,
				queryShape,
				1,
				2,
				3,
				4,
			)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			So(output.Location(), ShouldEqual, computetensor.Metal)
			assertMetalMaxDiff(values, referenceMQA(query, key, value, queryShape.Dims()), 1e-4)
		})
	})
}

func TestMetalAttention_SlidingWindowTensor(test *testing.T) {
	Convey("Given resident Metal sliding-window inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		attentionOps := attentionOpsForTest(test, tensorBackend)

		Convey("It should match CPU sliding-window attention", func() {
			shape, err := computetensor.NewShape([]int{1, 2, 4, 3})
			So(err, ShouldBeNil)

			query := positionalSequence(shape.Len())
			key := positionalSequence(shape.Len())
			value := positionalSequence(shape.Len())
			queryTensor := uploadMetalTensorForTest(test, tensorBackend, shape, query)
			keyTensor := uploadMetalTensorForTest(test, tensorBackend, shape, key)
			valueTensor := uploadMetalTensorForTest(test, tensorBackend, shape, value)

			output, err := attentionOps.SlidingWindowTensor(
				queryTensor,
				keyTensor,
				valueTensor,
				shape,
				1,
				2,
				4,
				3,
				1,
			)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			So(output.Location(), ShouldEqual, computetensor.Metal)
			assertMetalMaxDiff(values, referenceSlidingWindow(query, key, value, shape.Dims(), 1), 1e-4)
		})
	})
}

func TestTensorBackend_applyAttentionVariantGraphs(test *testing.T) {
	Convey("Given Metal attention variant graph execution", test, func() {
		Convey("It should execute MQA without intermediate readback", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			queryShape, keyValueShape := mqaShapes(test)
			query := positionalSequence(queryShape.Len())
			key := positionalSequence(keyValueShape.Len())
			value := positionalSequence(keyValueShape.Len())
			output := runAttentionVariantGraphForTest(
				test,
				tensorBackend,
				"attention.mqa",
				queryShape,
				keyValueShape,
				query,
				key,
				value,
				map[string]any{},
			)
			assertMetalMaxDiff(output, referenceMQA(query, key, value, queryShape.Dims()), 1e-4)
		})

		Convey("It should execute sliding-window attention without intermediate readback", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			shape, err := computetensor.NewShape([]int{1, 2, 4, 3})
			So(err, ShouldBeNil)

			query := positionalSequence(shape.Len())
			key := positionalSequence(shape.Len())
			value := positionalSequence(shape.Len())
			output := runAttentionVariantGraphForTest(
				test,
				tensorBackend,
				"attention.sliding_window",
				shape,
				shape,
				query,
				key,
				value,
				map[string]any{"window": 1},
			)
			assertMetalMaxDiff(output, referenceSlidingWindow(query, key, value, shape.Dims(), 1), 1e-4)
		})
	})
}

func BenchmarkMetalAttention_MQATensor(benchmark *testing.B) {
	benchmarkAttentionVariantTensor(benchmark, "mqa")
}

func BenchmarkMetalAttention_SlidingWindowTensor(benchmark *testing.B) {
	benchmarkAttentionVariantTensor(benchmark, "sliding_window")
}

func attentionOpsForTest(test testing.TB, tensorBackend *TensorBackend) *MetalAttention {
	test.Helper()

	attentionOps, err := tensorBackend.attention()
	So(err, ShouldBeNil)

	return attentionOps
}

func mqaShapes(test testing.TB) (computetensor.Shape, computetensor.Shape) {
	test.Helper()

	queryShape, err := computetensor.NewShape([]int{1, 2, 3, 4})
	So(err, ShouldBeNil)

	keyValueShape, err := computetensor.NewShape([]int{1, 1, 3, 4})
	So(err, ShouldBeNil)

	return queryShape, keyValueShape
}

func runAttentionVariantGraphForTest(
	test *testing.T,
	tensorBackend *TensorBackend,
	operationID ir.OpType,
	queryShape computetensor.Shape,
	keyValueShape computetensor.Shape,
	query []float64,
	key []float64,
	value []float64,
	metadata map[string]any,
) []float64 {
	test.Helper()

	queryNode := attentionInputNode("query", queryShape, query)
	keyNode := attentionInputNode("key", keyValueShape, key)
	valueNode := attentionInputNode("value", keyValueShape, value)
	outputNode := ir.NewNode("output", operationID, queryShape)
	outputNode.AddInput(queryNode)
	outputNode.AddInput(keyNode)
	outputNode.AddInput(valueNode)

	for key, value := range metadata {
		outputNode.SetMetadata(key, value)
	}

	graph := ir.NewGraph()
	graph.AddNode(queryNode)
	graph.AddNode(keyNode)
	graph.AddNode(valueNode)
	graph.AddNode(outputNode)

	before := tensorBackend.runtime.Metrics()
	results, err := NewRunnerWithBackend(tensorBackend).Execute(
		context.Background(),
		graph,
		[]*ir.Node{outputNode},
	)
	after := tensorBackend.runtime.Metrics()
	So(err, ShouldBeNil)
	So(results, ShouldHaveLength, 1)
	So(results["output"].Location(), ShouldEqual, computetensor.Metal)
	So(after.TransferBytes-before.TransferBytes, ShouldEqual, int64((len(query)+len(key)+len(value))*4))

	values, err := results["output"].CloneFloat64()
	So(err, ShouldBeNil)
	defer func() {
		So(results["output"].Close(), ShouldBeNil)
	}()

	return values
}

func attentionInputNode(id string, shape computetensor.Shape, values []float64) *ir.Node {
	node := ir.NewNode(id, ir.OpInput, shape)
	node.SetMetadata("values", values)

	return node
}

func referenceMQA(query []float64, key []float64, value []float64, shape []int) []float64 {
	stateDict := state.NewDict().WithShape(shape).WithInputs(query, key, value)
	expected, err := cpuattention.NewMQA().Forward(stateDict)
	So(err, ShouldBeNil)

	return expected.Out
}

func referenceSlidingWindow(
	query []float64,
	key []float64,
	value []float64,
	shape []int,
	window int,
) []float64 {
	stateDict := state.NewDict().WithShape(shape).WithInputs(query, key, value)
	stateDict.Window = window
	expected, err := cpuattention.NewSlidingWindow().Forward(stateDict)
	So(err, ShouldBeNil)

	return expected.Out
}

func benchmarkAttentionVariantTensor(benchmark *testing.B, variant string) {
	benchmark.ReportAllocs()

	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}
	defer func() {
		_ = tensorBackend.Close()
	}()

	attentionOps, err := tensorBackend.attention()
	if err != nil {
		benchmark.Fatal(err)
	}

	queryShape, err := computetensor.NewShape([]int{1, 8, 64, 32})
	if err != nil {
		benchmark.Fatal(err)
	}

	keyValueShape := queryShape
	if variant == "mqa" {
		keyValueShape, err = computetensor.NewShape([]int{1, 1, 64, 32})
		if err != nil {
			benchmark.Fatal(err)
		}
	}

	queryTensor := uploadMetalTensor(tensorBackend, queryShape, positionalSequence(queryShape.Len()))
	defer func() {
		_ = queryTensor.Close()
	}()

	keyTensor := uploadMetalTensor(tensorBackend, keyValueShape, positionalSequence(keyValueShape.Len()))
	defer func() {
		_ = keyTensor.Close()
	}()

	valueTensor := uploadMetalTensor(tensorBackend, keyValueShape, positionalSequence(keyValueShape.Len()))
	defer func() {
		_ = valueTensor.Close()
	}()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		output, err := benchmarkAttentionVariant(
			attentionOps,
			variant,
			queryTensor,
			keyTensor,
			valueTensor,
			queryShape,
		)
		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func benchmarkAttentionVariant(
	attentionOps *MetalAttention,
	variant string,
	queryTensor computetensor.Float64Tensor,
	keyTensor computetensor.Float64Tensor,
	valueTensor computetensor.Float64Tensor,
	queryShape computetensor.Shape,
) (computetensor.Float64Tensor, error) {
	if variant == "mqa" {
		return attentionOps.MQATensor(
			queryTensor,
			keyTensor,
			valueTensor,
			queryShape,
			1,
			8,
			64,
			32,
		)
	}

	return attentionOps.SlidingWindowTensor(
		queryTensor,
		keyTensor,
		valueTensor,
		queryShape,
		1,
		8,
		64,
		32,
		16,
	)
}
