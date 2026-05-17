//go:build darwin && cgo

package metal

import (
	"context"
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestMetalMetric_AccuracyTensor(test *testing.T) {
	Convey("Given resident Metal accuracy metric inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		mathOps := metalMathOpsForTest(test, tensorBackend)

		Convey("It should match the scalar accuracy reference at contract sizes", func() {
			for _, elementCount := range metalContractSizes() {
				predictions, targets := metalAccuracyValues(elementCount)
				output, err := mathOps.AccuracyTensor(
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, elementCount), predictions),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, elementCount), targets),
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, []float64{referenceAccuracy(predictions, targets)}, 0)
			}
		})
	})
}

func TestMetalMetric_PerplexityTensor(test *testing.T) {
	Convey("Given resident Metal perplexity metric inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		mathOps := metalMathOpsForTest(test, tensorBackend)

		Convey("It should match the scalar perplexity reference at contract sizes", func() {
			for _, elementCount := range metalContractSizes() {
				probabilities, targets := metalPerplexityValues(elementCount)
				output, err := mathOps.PerplexityTensor(
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, elementCount), probabilities),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, elementCount), targets),
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, []float64{referencePerplexity(probabilities, targets)}, 2e-6)
			}
		})
	})
}

func TestMetalMetric_F1Tensor(test *testing.T) {
	Convey("Given resident Metal F1 metric inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		mathOps := metalMathOpsForTest(test, tensorBackend)

		Convey("It should match the scalar F1 reference at contract sizes", func() {
			for _, elementCount := range metalContractSizes() {
				predictions, targets := metalF1Values(elementCount)
				output, err := mathOps.F1Tensor(
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, elementCount), predictions),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, elementCount), targets),
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, []float64{referenceF1(predictions, targets)}, 2e-6)
			}
		})
	})
}

func TestTensorBackend_applyMetricGraph(test *testing.T) {
	Convey("Given Metal metric graph execution", test, func() {
		Convey("It should keep metric graph outputs resident", func() {
			for _, scenario := range metricGraphScenarios(test) {
				tensorBackend := newMetalTensorBackendForTest(test)
				graph, target, expectedBytes := metricGraph(test, scenario)

				before := tensorBackend.runtime.Metrics()
				results, err := NewRunnerWithBackend(tensorBackend).Execute(context.Background(), graph, []*ir.Node{target})
				after := tensorBackend.runtime.Metrics()
				So(err, ShouldBeNil)
				So(results, ShouldHaveLength, 1)
				So(after.TransferBytes-before.TransferBytes, ShouldEqual, expectedBytes)

				output := results[target.ID()]
				So(output.Location(), ShouldEqual, computetensor.Metal)
				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				assertMetalMaxDiff(values, scenario.expected, scenario.tolerance)
				So(output.Close(), ShouldBeNil)
			}
		})
	})
}

func BenchmarkMetalMetric_AccuracyTensor(benchmark *testing.B) {
	benchmarkMetricTensor(benchmark, "accuracy")
}

func BenchmarkMetalMetric_PerplexityTensor(benchmark *testing.B) {
	benchmarkMetricTensor(benchmark, "perplexity")
}

func BenchmarkMetalMetric_F1Tensor(benchmark *testing.B) {
	benchmarkMetricTensor(benchmark, "f1")
}

type metricGraphScenario struct {
	operation string
	left      []float64
	right     []float64
	expected  []float64
	tolerance float64
}

func metricGraphScenarios(test testing.TB) []metricGraphScenario {
	test.Helper()

	accuracyPredictions, accuracyTargets := metalAccuracyValues(64)
	probabilities, probabilityTargets := metalPerplexityValues(64)
	f1Predictions, f1Targets := metalF1Values(64)

	return []metricGraphScenario{
		{"bench.accuracy", accuracyPredictions, accuracyTargets, []float64{referenceAccuracy(accuracyPredictions, accuracyTargets)}, 0},
		{"bench.metric.accuracy", accuracyPredictions, accuracyTargets, []float64{referenceAccuracy(accuracyPredictions, accuracyTargets)}, 0},
		{"bench.perplexity", probabilities, probabilityTargets, []float64{referencePerplexity(probabilities, probabilityTargets)}, 2e-6},
		{"bench.metric.perplexity", probabilities, probabilityTargets, []float64{referencePerplexity(probabilities, probabilityTargets)}, 2e-6},
		{"bench.f1", f1Predictions, f1Targets, []float64{referenceF1(f1Predictions, f1Targets)}, 2e-6},
		{"bench.metric.f1", f1Predictions, f1Targets, []float64{referenceF1(f1Predictions, f1Targets)}, 2e-6},
	}
}

func metricGraph(
	test testing.TB,
	scenario metricGraphScenario,
) (*ir.Graph, *ir.Node, int64) {
	test.Helper()

	values := [][]float64{scenario.left, scenario.right}
	names := []string{"metric_left", "metric_right"}
	graph := ir.NewGraph()
	inputs := make([]*ir.Node, len(values))
	expectedBytes := int64(0)

	for index, value := range values {
		node := ir.NewNode(names[index]+"_"+scenario.operation, ir.OpInput, causalShape(test, len(value)))
		node.SetMetadata("values", value)
		graph.AddNode(node)
		inputs[index] = node
		expectedBytes += int64(len(value) * 4)
	}

	target := ir.NewNode("metric_"+scenario.operation, ir.OpType(scenario.operation), causalShape(test, 1))
	for _, input := range inputs {
		target.AddInput(input)
	}
	graph.AddNode(target)

	return graph, target, expectedBytes
}

func benchmarkMetricTensor(benchmark *testing.B, operation string) {
	benchmark.ReportAllocs()
	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}
	defer func() {
		_ = tensorBackend.Close()
	}()

	mathOps, err := tensorBackend.math()
	if err != nil {
		benchmark.Fatal(err)
	}

	left, right := metalAccuracyValues(8192)
	if operation == "perplexity" {
		left, right = metalPerplexityValues(8192)
	}
	if operation == "f1" {
		left, right = metalF1Values(8192)
	}

	leftTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, len(left)), left)
	rightTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, len(right)), right)
	defer closeBenchmarkTensors(leftTensor, rightTensor)

	benchmark.ResetTimer()
	for benchmark.Loop() {
		var output computetensor.Float64Tensor
		switch operation {
		case "accuracy":
			output, err = mathOps.AccuracyTensor(leftTensor, rightTensor)
		case "perplexity":
			output, err = mathOps.PerplexityTensor(leftTensor, rightTensor)
		case "f1":
			output, err = mathOps.F1Tensor(leftTensor, rightTensor)
		}
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func metalAccuracyValues(elementCount int) ([]float64, []float64) {
	predictions := make([]float64, elementCount)
	targets := make([]float64, elementCount)
	winner := (elementCount * 3) / 5

	for index := range elementCount {
		predictions[index] = float64(float32(-0.8 + 0.001*float64(index%41)))
		targets[index] = float64(float32(-0.7 + 0.0013*float64(index%43)))
	}

	predictions[winner] = 2.0
	targets[winner] = 1.5

	return predictions, targets
}

func metalPerplexityValues(elementCount int) ([]float64, []float64) {
	probabilities := make([]float64, elementCount)
	targets := make([]float64, elementCount)
	targetIndex := elementCount / 2

	if elementCount == 1 {
		probabilities[0] = 1
		targets[0] = 1

		return probabilities, targets
	}

	background := float64(float32(0.75 / float64(elementCount-1)))
	for index := range probabilities {
		probabilities[index] = background
	}
	probabilities[targetIndex] = 0.25
	targets[targetIndex] = 1

	return probabilities, targets
}

func metalF1Values(elementCount int) ([]float64, []float64) {
	predictions := make([]float64, elementCount)
	targets := make([]float64, elementCount)

	for index := range elementCount {
		if index%4 < 2 {
			predictions[index] = 0.8
		}
		if index%4 >= 2 {
			predictions[index] = 0.2
		}
		if index%3 == 0 {
			targets[index] = 0.7
		}
		if index%3 != 0 {
			targets[index] = 0.1
		}
	}

	return predictions, targets
}

func referenceAccuracy(predictions []float64, targets []float64) float64 {
	if argmaxFloat64(predictions) == argmaxFloat64(targets) {
		return 1
	}

	return 0
}

func referencePerplexity(probabilities []float64, targets []float64) float64 {
	loss := float32(0)

	for index, target := range targets {
		loss += float32(-math.Log(probabilities[index]+1e-9) * target)
	}

	return float64(float32(math.Exp(float64(loss))))
}

func referenceF1(predictions []float64, targets []float64) float64 {
	truePositive := float32(0)
	falsePositive := float32(0)
	falseNegative := float32(0)

	for index := range predictions {
		predicted := predictions[index] >= 0.5
		actual := targets[index] >= 0.5

		if predicted && actual {
			truePositive++
		}
		if predicted && !actual {
			falsePositive++
		}
		if !predicted && actual {
			falseNegative++
		}
	}

	epsilon := float32(1e-9)
	precision := truePositive / (truePositive + falsePositive + epsilon)
	recall := truePositive / (truePositive + falseNegative + epsilon)

	return float64(float32(2 * precision * recall / (precision + recall + epsilon)))
}

func argmaxFloat64(values []float64) int {
	bestIndex := 0
	bestValue := values[0]

	for index, value := range values[1:] {
		if value <= bestValue {
			continue
		}

		bestIndex = index + 1
		bestValue = value
	}

	return bestIndex
}
