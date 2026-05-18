//go:build darwin && cgo

package metal

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestProjectionOps_FusedQKVTensor(test *testing.T) {
	Convey("Given resident Metal fused QKV inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		projectionOps := projectionOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar projection reference", func() {
			rows, inFeatures, outFeatures := 2, 3, 6
			inputShape, outputShape := fusedQKVShapes(test, rows, inFeatures, outFeatures)
			input := fusedQKVInput(rows * inFeatures)
			weight := fusedQKVWeight(inFeatures * outFeatures)
			bias := fusedQKVBias(outFeatures)
			inputTensor := uploadMetalTensorForTest(test, tensorBackend, inputShape, input)
			weightTensor := uploadMetalTensorForTest(test, tensorBackend, weightShape(test, inFeatures, outFeatures), weight)
			biasTensor := uploadMetalTensorForTest(test, tensorBackend, biasShape(test, outFeatures), bias)

			output, err := projectionOps.FusedQKVTensor(
				inputTensor,
				weightTensor,
				biasTensor,
				outputShape,
				rows,
				inFeatures,
				outFeatures,
			)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := tensorFloat64Values(output)
			So(err, ShouldBeNil)
			So(output.Location(), ShouldEqual, computetensor.Metal)
			assertMetalMaxDiff(
				values,
				referenceFusedQKV(input, weight, bias, rows, inFeatures, outFeatures),
				1e-5,
			)
		})
	})
}

func TestTensorBackend_applyFusedQKVGraph(test *testing.T) {
	Convey("Given Metal fused QKV graph execution", test, func() {
		Convey("It should execute with persistent resident parameters", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			rows, inFeatures, outFeatures := 2, 3, 6
			inputShape, outputShape := fusedQKVShapes(test, rows, inFeatures, outFeatures)
			input := fusedQKVInput(rows * inFeatures)
			weight := fusedQKVWeight(inFeatures * outFeatures)
			bias := fusedQKVBias(outFeatures)
			inputNode := ir.NewNode("input", ir.OpInput, inputShape)
			inputNode.SetMetadata("values", input)
			outputNode := ir.NewNode("qkv", "projection.fused_qkv", outputShape)
			outputNode.SetMetadata("d_in", inFeatures)
			outputNode.SetMetadata("d_q", 2)
			outputNode.SetMetadata("d_k", 2)
			outputNode.SetMetadata("d_v", 2)
			outputNode.SetMetadata("weight", weight)
			outputNode.SetMetadata("bias", bias)
			outputNode.AddInput(inputNode)

			graph := ir.NewGraph()
			graph.AddNode(inputNode)
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
			So(results["qkv"].Location(), ShouldEqual, computetensor.Metal)
			So(after.TransferBytes-before.TransferBytes, ShouldEqual, int64((len(input)+len(weight)+len(bias))*4))

			values, err := tensorFloat64Values(results["qkv"])
			So(err, ShouldBeNil)
			defer func() {
				So(results["qkv"].Close(), ShouldBeNil)
			}()
			assertMetalMaxDiff(
				values,
				referenceFusedQKV(input, weight, bias, rows, inFeatures, outFeatures),
				1e-5,
			)
		})
	})
}

func BenchmarkProjectionOps_FusedQKVTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()

	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}
	defer func() {
		_ = tensorBackend.Close()
	}()

	projectionOps, err := tensorBackend.projection()
	if err != nil {
		benchmark.Fatal(err)
	}

	rows, inFeatures, outFeatures := 64, 128, 384
	inputShape, outputShape := benchmarkFusedQKVShapes(benchmark, rows, inFeatures, outFeatures)
	inputTensor := uploadMetalTensor(tensorBackend, inputShape, fusedQKVInput(rows*inFeatures))
	defer func() {
		_ = inputTensor.Close()
	}()

	weightTensor := uploadMetalTensor(
		tensorBackend,
		benchmarkWeightShape(benchmark, inFeatures, outFeatures),
		fusedQKVWeight(inFeatures*outFeatures),
	)
	defer func() {
		_ = weightTensor.Close()
	}()

	biasTensor := uploadMetalTensor(
		tensorBackend,
		benchmarkBiasShape(benchmark, outFeatures),
		fusedQKVBias(outFeatures),
	)
	defer func() {
		_ = biasTensor.Close()
	}()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		output, err := projectionOps.FusedQKVTensor(
			inputTensor,
			weightTensor,
			biasTensor,
			outputShape,
			rows,
			inFeatures,
			outFeatures,
		)
		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func projectionOpsForTest(test testing.TB, tensorBackend *TensorBackend) *ProjectionOps {
	test.Helper()

	projectionOps, err := tensorBackend.projection()
	So(err, ShouldBeNil)

	return projectionOps
}

func fusedQKVShapes(
	test testing.TB,
	rows int,
	inFeatures int,
	outFeatures int,
) (computetensor.Shape, computetensor.Shape) {
	test.Helper()

	inputShape, err := computetensor.NewShape([]int{rows, inFeatures})
	So(err, ShouldBeNil)

	outputShape, err := computetensor.NewShape([]int{rows, outFeatures})
	So(err, ShouldBeNil)

	return inputShape, outputShape
}

func benchmarkFusedQKVShapes(
	benchmark *testing.B,
	rows int,
	inFeatures int,
	outFeatures int,
) (computetensor.Shape, computetensor.Shape) {
	benchmark.Helper()

	inputShape, err := computetensor.NewShape([]int{rows, inFeatures})
	if err != nil {
		benchmark.Fatal(err)
	}

	outputShape, err := computetensor.NewShape([]int{rows, outFeatures})
	if err != nil {
		benchmark.Fatal(err)
	}

	return inputShape, outputShape
}

func weightShape(test testing.TB, inFeatures int, outFeatures int) computetensor.Shape {
	test.Helper()

	shape, err := computetensor.NewShape([]int{inFeatures, outFeatures})
	So(err, ShouldBeNil)

	return shape
}

func biasShape(test testing.TB, outFeatures int) computetensor.Shape {
	test.Helper()

	shape, err := computetensor.NewShape([]int{outFeatures})
	So(err, ShouldBeNil)

	return shape
}

func benchmarkWeightShape(
	benchmark *testing.B,
	inFeatures int,
	outFeatures int,
) computetensor.Shape {
	benchmark.Helper()

	shape, err := computetensor.NewShape([]int{inFeatures, outFeatures})
	if err != nil {
		benchmark.Fatal(err)
	}

	return shape
}

func benchmarkBiasShape(benchmark *testing.B, outFeatures int) computetensor.Shape {
	benchmark.Helper()

	shape, err := computetensor.NewShape([]int{outFeatures})
	if err != nil {
		benchmark.Fatal(err)
	}

	return shape
}

func fusedQKVInput(length int) []float64 {
	values := make([]float64, length)

	for index := range values {
		values[index] = float64((index%13)-6) / 7
	}

	return values
}

func fusedQKVWeight(length int) []float64 {
	values := make([]float64, length)

	for index := range values {
		values[index] = float64((index%17)-8) / 11
	}

	return values
}

func fusedQKVBias(length int) []float64 {
	values := make([]float64, length)

	for index := range values {
		values[index] = float64((index%5)-2) / 9
	}

	return values
}

func referenceFusedQKV(
	input []float64,
	weight []float64,
	bias []float64,
	rows int,
	inFeatures int,
	outFeatures int,
) []float64 {
	output := make([]float64, rows*outFeatures)

	for rowIndex := range rows {
		for columnIndex := range outFeatures {
			sum := float32(0)

			for innerIndex := range inFeatures {
				left := float32(input[rowIndex*inFeatures+innerIndex])
				right := float32(weight[innerIndex*outFeatures+columnIndex])
				sum += left * right
			}

			if len(bias) != 0 {
				sum += float32(bias[columnIndex])
			}

			output[rowIndex*outFeatures+columnIndex] = float64(sum)
		}
	}

	return output
}
