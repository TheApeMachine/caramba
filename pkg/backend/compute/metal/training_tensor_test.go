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

func TestMetalTraining_MSELossTensor(test *testing.T) {
	Convey("Given resident Metal MSE loss inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		mathOps := metalMathOpsForTest(test, tensorBackend)

		Convey("It should match the scalar MSE loss reference at contract sizes", func() {
			for _, elementCount := range metalContractSizes() {
				predictions, targets := metalTrainingValues(elementCount)
				output, err := mathOps.MSELossTensor(
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, elementCount), predictions),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, elementCount), targets),
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := tensorFloat64Values(output)
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, []float64{referenceMSELoss(predictions, targets)}, 2e-5)
			}
		})
	})
}

func TestMetalTraining_CrossEntropyLossTensor(test *testing.T) {
	Convey("Given resident Metal cross entropy loss inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		mathOps := metalMathOpsForTest(test, tensorBackend)

		Convey("It should match the scalar cross entropy loss reference at contract sizes", func() {
			for _, elementCount := range metalContractSizes() {
				logits, targets := metalCrossEntropyValues(elementCount)
				output, err := mathOps.CrossEntropyLossTensor(
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, elementCount), logits),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, elementCount), targets),
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := tensorFloat64Values(output)
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, []float64{referenceCrossEntropyLoss(logits, targets)}, 2e-5)
			}
		})
	})
}

func TestMetalTraining_MSEGradTensor(test *testing.T) {
	Convey("Given resident Metal MSE gradient inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		mathOps := metalMathOpsForTest(test, tensorBackend)

		Convey("It should match the scalar MSE gradient reference at contract sizes", func() {
			for _, elementCount := range metalContractSizes() {
				predictions, targets := metalTrainingValues(elementCount)
				output, err := mathOps.MSEGradTensor(
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, elementCount), predictions),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, elementCount), targets),
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := tensorFloat64Values(output)
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, referenceMSEGrad(predictions, targets), 2e-6)
			}
		})
	})
}

func TestMetalTraining_CrossEntropyGradTensor(test *testing.T) {
	Convey("Given resident Metal cross entropy gradient inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		mathOps := metalMathOpsForTest(test, tensorBackend)

		Convey("It should match the scalar cross entropy gradient reference at contract sizes", func() {
			for _, elementCount := range metalContractSizes() {
				logits, targets := metalCrossEntropyValues(elementCount)
				output, err := mathOps.CrossEntropyGradTensor(
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, elementCount), logits),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, elementCount), targets),
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := tensorFloat64Values(output)
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, referenceCrossEntropyGrad(logits, targets), 2e-5)
			}
		})
	})
}

func TestTensorBackend_applyTrainingGraph(test *testing.T) {
	Convey("Given Metal training graph execution", test, func() {
		Convey("It should keep training graph outputs resident", func() {
			for _, scenario := range trainingGraphScenarios(test) {
				tensorBackend := newMetalTensorBackendForTest(test)
				graph, target, expectedBytes := trainingGraph(test, scenario)

				before := tensorBackend.runtime.Metrics()
				results, err := NewRunnerWithBackend(tensorBackend).Execute(context.Background(), graph, []*ir.Node{target})
				after := tensorBackend.runtime.Metrics()
				So(err, ShouldBeNil)
				So(results, ShouldHaveLength, 1)
				So(after.TransferBytes-before.TransferBytes, ShouldEqual, expectedBytes)

				output := results[target.ID()]
				So(output.Location(), ShouldEqual, computetensor.Metal)
				values, err := tensorFloat64Values(output)
				So(err, ShouldBeNil)
				assertMetalMaxDiff(values, scenario.expected, scenario.tolerance)
				So(output.Close(), ShouldBeNil)
			}
		})
	})
}

func BenchmarkMetalTraining_MSELossTensor(benchmark *testing.B) {
	benchmarkTrainingTensor(benchmark, "mse_loss")
}

func BenchmarkMetalTraining_CrossEntropyLossTensor(benchmark *testing.B) {
	benchmarkTrainingTensor(benchmark, "cross_entropy_loss")
}

func BenchmarkMetalTraining_MSEGradTensor(benchmark *testing.B) {
	benchmarkTrainingTensor(benchmark, "mse_grad")
}

func BenchmarkMetalTraining_CrossEntropyGradTensor(benchmark *testing.B) {
	benchmarkTrainingTensor(benchmark, "cross_entropy_grad")
}

type trainingGraphScenario struct {
	operation string
	left      []float64
	right     []float64
	expected  []float64
	tolerance float64
}

func trainingGraphScenarios(test testing.TB) []trainingGraphScenario {
	test.Helper()

	predictions, targets := metalTrainingValues(64)
	logits, classes := metalCrossEntropyValues(64)

	return []trainingGraphScenario{
		{"train.loss.mse", predictions, targets, []float64{referenceMSELoss(predictions, targets)}, 2e-5},
		{"train.loss.cross_entropy", logits, classes, []float64{referenceCrossEntropyLoss(logits, classes)}, 2e-5},
		{"train.loss.mse_grad", predictions, targets, referenceMSEGrad(predictions, targets), 2e-6},
		{"train.loss.cross_entropy_grad", logits, classes, referenceCrossEntropyGrad(logits, classes), 2e-5},
	}
}

func trainingGraph(
	test testing.TB,
	scenario trainingGraphScenario,
) (*ir.Graph, *ir.Node, int64) {
	test.Helper()

	values := [][]float64{scenario.left, scenario.right}
	names := []string{"training_left", "training_right"}
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

	target := ir.NewNode("training_"+scenario.operation, ir.OpType(scenario.operation), causalShape(test, len(scenario.expected)))
	for _, input := range inputs {
		target.AddInput(input)
	}
	graph.AddNode(target)

	return graph, target, expectedBytes
}

func benchmarkTrainingTensor(benchmark *testing.B, operation string) {
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

	left, right := metalTrainingValues(8192)
	if operation == "cross_entropy_loss" || operation == "cross_entropy_grad" {
		left, right = metalCrossEntropyValues(8192)
	}

	leftTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, len(left)), left)
	rightTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, len(right)), right)
	defer closeBenchmarkTensors(leftTensor, rightTensor)

	benchmark.ResetTimer()
	for benchmark.Loop() {
		var output computetensor.Tensor
		switch operation {
		case "mse_loss":
			output, err = mathOps.MSELossTensor(leftTensor, rightTensor)
		case "cross_entropy_loss":
			output, err = mathOps.CrossEntropyLossTensor(leftTensor, rightTensor)
		case "mse_grad":
			output, err = mathOps.MSEGradTensor(leftTensor, rightTensor)
		case "cross_entropy_grad":
			output, err = mathOps.CrossEntropyGradTensor(leftTensor, rightTensor)
		}
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func metalTrainingValues(elementCount int) ([]float64, []float64) {
	predictions := make([]float64, elementCount)
	targets := make([]float64, elementCount)

	for index := range elementCount {
		predictions[index] = float64(float32(-0.4 + 0.013*float64(index%31)))
		targets[index] = float64(float32(0.2 - 0.011*float64(index%29)))
	}

	return predictions, targets
}

func metalCrossEntropyValues(elementCount int) ([]float64, []float64) {
	logits := make([]float64, elementCount)
	targets := make([]float64, elementCount)

	for index := range elementCount {
		logits[index] = float64(float32(-0.3 + 0.017*float64(index%37)))
	}
	targets[elementCount/2] = 1

	return logits, targets
}

func referenceMSELoss(predictions []float64, targets []float64) float64 {
	sum := 0.0

	for index, prediction := range predictions {
		diff := prediction - targets[index]
		sum += diff * diff
	}

	return float64(float32(sum / float64(len(predictions))))
}

func referenceMSEGrad(predictions []float64, targets []float64) []float64 {
	values := make([]float64, len(predictions))
	scale := float64(float32(2 / float64(len(predictions))))

	for index, prediction := range predictions {
		values[index] = float64(float32((prediction - targets[index]) * scale))
	}

	return values
}

func referenceCrossEntropyLoss(logits []float64, targets []float64) float64 {
	probabilities := referenceTrainingSoftmax(logits)
	loss := 0.0

	for index, target := range targets {
		loss -= math.Log(probabilities[index]+1e-9) * target
	}

	return float64(float32(loss))
}

func referenceCrossEntropyGrad(logits []float64, targets []float64) []float64 {
	probabilities := referenceTrainingSoftmax(logits)

	for index := range probabilities {
		probabilities[index] = float64(float32(probabilities[index] - targets[index]))
	}

	return probabilities
}

func referenceTrainingSoftmax(logits []float64) []float64 {
	maxValue := logits[0]
	for _, value := range logits[1:] {
		if value > maxValue {
			maxValue = value
		}
	}

	sum := 0.0
	probabilities := make([]float64, len(logits))
	for index, value := range logits {
		probability := math.Exp(value - maxValue)
		probabilities[index] = probability
		sum += probability
	}

	for index := range probabilities {
		probabilities[index] = float64(float32(probabilities[index] / sum))
	}

	return probabilities
}
