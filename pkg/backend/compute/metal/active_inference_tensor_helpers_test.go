//go:build darwin && cgo

package metal

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func activeBenchmarkOps(benchmark *testing.B) (*TensorBackend, *ActiveInferenceOps) {
	benchmark.Helper()

	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}

	activeOps, err := tensorBackend.activeInference()
	if err != nil {
		benchmark.Fatal(err)
	}

	return tensorBackend, activeOps
}

func activeShape(test testing.TB, elementCount int) computetensor.Shape {
	test.Helper()

	shape, err := computetensor.NewShape([]int{elementCount})
	So(err, ShouldBeNil)

	return shape
}

func activeBenchmarkShape(benchmark *testing.B, elementCount int) computetensor.Shape {
	benchmark.Helper()

	shape, err := computetensor.NewShape([]int{elementCount})
	if err != nil {
		benchmark.Fatal(err)
	}

	return shape
}

func referenceActiveFreeEnergy(mean []float64, logSigma []float64) float64 {
	sum := float32(0)

	for index := range mean {
		meanValue := float32(mean[index])
		logSigmaValue := float32(logSigma[index])
		sum += 0.5 * (meanValue*meanValue + float32(math.Exp(float64(logSigmaValue))) - logSigmaValue - 1)
	}

	return float64(sum)
}

func referenceActiveBeliefUpdate(
	mean []float64,
	logSigma []float64,
	predictionError []float64,
	learningRate float32,
) []float64 {
	values := make([]float64, 2*len(mean))

	for index := range mean {
		meanValue := float32(mean[index])
		logSigmaValue := float32(logSigma[index])
		errorValue := float32(predictionError[index])
		values[index] = float64(meanValue - learningRate*(meanValue+errorValue))
		values[len(mean)+index] = float64(
			logSigmaValue - learningRate*(float32(math.Exp(float64(logSigmaValue)))-1),
		)
	}

	return values
}

func referenceActivePrecisionWeight(predictionError []float64, logPrecision []float64) []float64 {
	values := make([]float64, len(predictionError))

	for index := range values {
		logPrecisionValue := float32(logPrecision[index])
		logPrecisionValue = min(max(logPrecisionValue, -80), 80)
		values[index] = float64(
			float32(predictionError[index]) *
				float32(math.Exp(float64(logPrecisionValue))),
		)
	}

	return values
}

func referenceActiveExpectedFreeEnergy(
	probabilities []float64,
	outcomeCount int,
	policyCount int,
) []float64 {
	values := make([]float64, policyCount)
	eps := float32(DefaultExpectedFreeEnergyEps)

	for policyIndex := range policyCount {
		accumulator := float32(0)

		for outcomeIndex := range outcomeCount {
			probability := float32(probabilities[outcomeIndex*policyCount+policyIndex])
			probability = min(max(probability, 0), 1)
			accumulator -= probability * float32(math.Log(float64(probability+eps)))
		}

		values[policyIndex] = float64(accumulator)
	}

	return values
}

func activeInferenceGraph(test testing.TB) (
	*ir.Graph,
	[]*ir.Node,
	int64,
	map[string][]float64,
) {
	test.Helper()

	dimension, policyCount := 64, 3
	mean, logSigma := activeInferenceGaussianInputs(dimension)
	predictionError := activeInferenceSequence(dimension, 0.019, -0.14)
	probabilities := activeInferenceOutcomeInputs(dimension, policyCount)
	inputs := activeGraphInputs(test, mean, logSigma, predictionError, probabilities)

	freeEnergyNode := activeNode("active_free", "active_inference.free_energy", activeShape(test, 1), inputs[0], inputs[1])
	beliefNode := activeNode("active_belief", "active_inference.belief_update", activeShape(test, 2*dimension), inputs[0], inputs[1], inputs[2])
	beliefNode.SetMetadata("learning_rate", 0.0125)
	precisionNode := activeNode("active_precision", "active_inference.precision_weight", activeShape(test, dimension), inputs[2], inputs[1])
	expectedNode := activeNode("active_expected", "active_inference.expected_free_energy", activeShape(test, policyCount), inputs[3])
	expectedNode.SetMetadata("outcome_count", dimension)

	graph := ir.NewGraph()
	for _, node := range append(inputs, freeEnergyNode, beliefNode, precisionNode, expectedNode) {
		graph.AddNode(node)
	}

	expected := map[string][]float64{
		"active_free":      []float64{referenceActiveFreeEnergy(mean, logSigma)},
		"active_belief":    referenceActiveBeliefUpdate(mean, logSigma, predictionError, 0.0125),
		"active_precision": referenceActivePrecisionWeight(predictionError, logSigma),
		"active_expected":  referenceActiveExpectedFreeEnergy(probabilities, dimension, policyCount),
	}
	expectedBytes := int64((len(mean) + len(logSigma) + len(predictionError) + len(probabilities)) * 4)

	return graph, []*ir.Node{freeEnergyNode, beliefNode, precisionNode, expectedNode}, expectedBytes, expected
}

func activeGraphInputs(test testing.TB, values ...[]float64) []*ir.Node {
	test.Helper()

	names := []string{"active_mean", "active_log_sigma", "active_error", "active_probabilities"}
	nodes := make([]*ir.Node, len(values))

	for index, value := range values {
		node := ir.NewNode(names[index], ir.OpInput, activeShape(test, len(value)))
		node.SetMetadata("values", value)
		nodes[index] = node
	}

	return nodes
}

func activeNode(
	name string,
	operation ir.OpType,
	shape computetensor.Shape,
	inputs ...*ir.Node,
) *ir.Node {
	node := ir.NewNode(name, operation, shape)

	for _, input := range inputs {
		node.AddInput(input)
	}

	return node
}

func assertActiveInferenceGraphOutputs(
	results map[string]computetensor.Tensor,
	expected map[string][]float64,
) {
	for name, output := range results {
		So(output.Location(), ShouldEqual, computetensor.Metal)
		defer func(value computetensor.Tensor) {
			So(value.Close(), ShouldBeNil)
		}(output)

		values, err := tensorFloat64Values(output)
		So(err, ShouldBeNil)
		assertMetalMaxDiff(values, expected[name], activeGraphTolerance(name))
	}
}

func activeGraphTolerance(name string) float64 {
	if name == "active_free" || name == "active_expected" {
		return 1e-4
	}

	return 1e-5
}
