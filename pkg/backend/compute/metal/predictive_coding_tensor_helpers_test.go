//go:build darwin && cgo

package metal

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func predictiveShape(test testing.TB, elementCount int) computetensor.Shape {
	test.Helper()

	shape, err := computetensor.NewShape([]int{elementCount})
	if err != nil {
		test.Fatalf("predictive shape: %v", err)
	}

	return shape
}

func predictiveVector(elementCount int, scale float64, offset float64) []float64 {
	values := make([]float64, elementCount)

	for index := range values {
		values[index] = float64(float32(offset + scale*float64(index%19-9)))
	}

	return values
}

func predictiveWeights(outFeatures int, inFeatures int) []float64 {
	values := make([]float64, outFeatures*inFeatures)

	for index := range values {
		values[index] = float64(float32(0.01 * float64(index%17-8)))
	}

	return values
}

func referencePredictivePrediction(weights []float64, representation []float64, outFeatures int, inFeatures int) []float64 {
	values := make([]float64, outFeatures)

	for outIndex := range outFeatures {
		sum := float32(0)
		for inIndex := range inFeatures {
			sum += float32(weights[outIndex*inFeatures+inIndex]) * float32(representation[inIndex])
		}

		values[outIndex] = float64(sum)
	}

	return values
}

func referencePredictiveError(observation []float64, prediction []float64, precision []float64) []float64 {
	values := make([]float64, len(observation))

	for index := range values {
		errorValue := float32(observation[index]) - float32(prediction[index])
		if precision != nil {
			errorValue *= float32(precision[index])
		}

		values[index] = float64(errorValue)
	}

	return values
}

func referencePredictiveUpdateRepresentation(
	representation []float64,
	weights []float64,
	lowerError []float64,
	selfError []float64,
	learningRate float64,
	outFeatures int,
	inFeatures int,
) []float64 {
	values := make([]float64, inFeatures)

	for inIndex := range inFeatures {
		accumulator := -float32(selfError[inIndex])
		for outIndex := range outFeatures {
			accumulator += float32(weights[outIndex*inFeatures+inIndex]) * float32(lowerError[outIndex])
		}

		values[inIndex] = float64(float32(representation[inIndex]) + float32(learningRate)*accumulator)
	}

	return values
}

func referencePredictiveUpdateWeights(
	weights []float64,
	errorValues []float64,
	representation []float64,
	learningRate float64,
	outFeatures int,
	inFeatures int,
) []float64 {
	values := make([]float64, len(weights))

	for outIndex := range outFeatures {
		for inIndex := range inFeatures {
			offset := outIndex*inFeatures + inIndex
			values[offset] = float64(
				float32(weights[offset]) +
					float32(learningRate)*float32(errorValues[outIndex])*float32(representation[inIndex]),
			)
		}
	}

	return values
}

func predictiveGraph(test testing.TB) (*ir.Graph, []*ir.Node, int64, map[string][]float64) {
	test.Helper()

	inFeatures, outFeatures := 64, 3
	weights := predictiveWeights(outFeatures, inFeatures)
	representation := predictiveVector(inFeatures, 0.013, -0.2)
	observation := predictiveVector(outFeatures, 0.011, 0.3)
	precision := predictiveVector(outFeatures, 0.003, 0.8)
	lowerError := predictiveVector(outFeatures, 0.017, -0.05)
	selfError := predictiveVector(inFeatures, 0.007, 0.04)
	learningRate := []float64{0.025}
	inputs := predictiveGraphInputs(test, weights, representation, observation, precision, lowerError, selfError, learningRate)

	predictionNode := predictiveNode("pc_prediction", "predictive_coding.prediction", predictiveShape(test, outFeatures), inputs[0], inputs[1])
	errorNode := predictiveNode("pc_error", "predictive_coding.prediction_error", predictiveShape(test, outFeatures), inputs[2], predictionNode, inputs[3])
	representationNode := predictiveNode("pc_representation", "predictive_coding.update_representation", predictiveShape(test, inFeatures), inputs[1], inputs[0], inputs[4], inputs[5], inputs[6])
	weightsNode := predictiveNode("pc_weights", "predictive_coding.update_weights", predictiveShape(test, len(weights)), inputs[0], inputs[4], inputs[1], inputs[6])
	graph := ir.NewGraph()

	for _, node := range append(inputs, predictionNode, errorNode, representationNode, weightsNode) {
		graph.AddNode(node)
	}

	prediction := referencePredictivePrediction(weights, representation, outFeatures, inFeatures)
	expected := map[string][]float64{
		"pc_prediction":     prediction,
		"pc_error":          referencePredictiveError(observation, prediction, precision),
		"pc_representation": referencePredictiveUpdateRepresentation(representation, weights, lowerError, selfError, learningRate[0], outFeatures, inFeatures),
		"pc_weights":        referencePredictiveUpdateWeights(weights, lowerError, representation, learningRate[0], outFeatures, inFeatures),
	}
	expectedBytes := int64(
		(len(weights) + len(representation) + len(observation) + len(precision) +
			len(lowerError) + len(selfError) + len(learningRate)) * 4,
	)

	return graph, []*ir.Node{predictionNode, errorNode, representationNode, weightsNode}, expectedBytes, expected
}

func predictiveGraphInputs(test testing.TB, values ...[]float64) []*ir.Node {
	test.Helper()

	names := []string{"pc_weights_input", "pc_representation_input", "pc_observation", "pc_precision", "pc_lower_error", "pc_self_error", "pc_lr"}
	nodes := make([]*ir.Node, len(values))

	for index, value := range values {
		node := ir.NewNode(names[index], ir.OpInput, predictiveShape(test, len(value)))
		node.SetMetadata("values", value)
		nodes[index] = node
	}

	return nodes
}

func predictiveNode(name string, operation ir.OpType, shape computetensor.Shape, inputs ...*ir.Node) *ir.Node {
	node := ir.NewNode(name, operation, shape)

	for _, input := range inputs {
		node.AddInput(input)
	}

	return node
}

func assertPredictiveGraphOutputs(results map[string]computetensor.Tensor, expected map[string][]float64) {
	for name, output := range results {
		So(output.Location(), ShouldEqual, computetensor.Metal)
		defer func(value computetensor.Tensor) {
			So(value.Close(), ShouldBeNil)
		}(output)

		values, err := tensorFloat64Values(output)
		So(err, ShouldBeNil)
		assertMetalMaxDiff(values, expected[name], 1e-4)
	}
}

func benchmarkPredictive(benchmark *testing.B, operation string) {
	benchmark.ReportAllocs()
	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}
	defer func() {
		_ = tensorBackend.Close()
	}()

	predictiveOps, err := tensorBackend.predictiveCoding()
	if err != nil {
		benchmark.Fatal(err)
	}

	predictiveBenchmarkLoop(benchmark, tensorBackend, predictiveOps, operation)
}

func predictiveBenchmarkLoop(
	benchmark *testing.B,
	tensorBackend *TensorBackend,
	predictiveOps *MetalPredictiveCodingOps,
	operation string,
) {
	inFeatures, outFeatures := 8192, 3
	weights := uploadMetalTensor(tensorBackend, predictiveShape(benchmark, outFeatures*inFeatures), predictiveWeights(outFeatures, inFeatures))
	representation := uploadMetalTensor(tensorBackend, predictiveShape(benchmark, inFeatures), predictiveVector(inFeatures, 0.013, -0.2))
	observation := uploadMetalTensor(tensorBackend, predictiveShape(benchmark, inFeatures), predictiveVector(inFeatures, 0.011, 0.3))
	prediction := uploadMetalTensor(tensorBackend, predictiveShape(benchmark, inFeatures), predictiveVector(inFeatures, 0.009, -0.1))
	precision := uploadMetalTensor(tensorBackend, predictiveShape(benchmark, inFeatures), predictiveVector(inFeatures, 0.003, 0.8))
	lowerError := uploadMetalTensor(tensorBackend, predictiveShape(benchmark, outFeatures), predictiveVector(outFeatures, 0.017, -0.05))
	selfError := uploadMetalTensor(tensorBackend, predictiveShape(benchmark, inFeatures), predictiveVector(inFeatures, 0.007, 0.04))
	learningRate := uploadMetalTensor(tensorBackend, predictiveShape(benchmark, 1), []float64{0.025})
	defer closeBenchmarkTensors(weights, representation, observation, prediction, precision, lowerError, selfError, learningRate)

	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := predictiveBenchmarkOutput(
			predictiveOps,
			operation,
			weights,
			representation,
			observation,
			prediction,
			precision,
			lowerError,
			selfError,
			learningRate,
			predictiveShape(benchmark, outFeatures),
		)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func predictiveBenchmarkOutput(
	predictiveOps *MetalPredictiveCodingOps,
	operation string,
	weights computetensor.Tensor,
	representation computetensor.Tensor,
	observation computetensor.Tensor,
	prediction computetensor.Tensor,
	precision computetensor.Tensor,
	lowerError computetensor.Tensor,
	selfError computetensor.Tensor,
	learningRate computetensor.Tensor,
	predictionShape computetensor.Shape,
) (computetensor.Tensor, error) {
	switch operation {
	case "prediction":
		return predictiveOps.PredictionTensor(weights, representation, predictionShape)
	case "prediction_error":
		return predictiveOps.PredictionErrorTensor(observation, prediction, precision)
	case "update_representation":
		return predictiveOps.UpdateRepresentationTensor(representation, weights, lowerError, selfError, learningRate)
	default:
		return predictiveOps.UpdateWeightsTensor(weights, lowerError, representation, weights.Shape(), learningRate)
	}
}
